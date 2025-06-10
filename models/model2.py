import torch.nn as nn
import torch
import random
from transformers import ViTImageProcessor, ViTForImageClassification
from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F
from modules.models import UNetWrapper, EmbeddingAdapter

from modules.models_t4 import BasicUpdateBlockDepth
# from newcrfs.networks.swin_transformer import SwinTransformer
# from newcrfs.networks.newcrf_layers import NewCRF
# from newcrfs.networks.uper_crf_head import PSP
from modules.xformer_decoder import TransformerDecoder, TransformerDecoder2

from transformers import logging
logging.set_verbosity_error()




def debug_mode(s, x=None):
    print('---------------debug--------------')
      
    if s[:5] == 'shape':
        print(s, x.shape)
    elif s == 'direct':
        print(x)
    elif s == 'list':
        for c in x:
            print(c.shape)
    elif s >= '0' and s<='9':
        print(s)

    print('---------------debug--------------')

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



class EcoDepthEncoder(nn.Module):
    def __init__(self, out_dim=1024, ldm_prior=[320, 640, 1280+1280], sd_path=None, emb_dim=768,
                 dataset='nyu', args=None):
        super().__init__()

        self.args = args

        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )
        self.apply(self._init_weights)
        
        self.cide_module = CIDE(args, emb_dim)
        
        self.config = OmegaConf.load('./v1-inference.yaml')
        if sd_path is None:
            self.config.model.params.ckpt_path = '/ossfs/workspace/ts/wangjinhong/code/Ecodepth/checkpoints/v1-5-pruned-emaonly.ckpt'
        else:
            self.config.model.params.ckpt_path = f'../{sd_path}'

        sd_model = instantiate_from_config(self.config.model)
        self.encoder_vq = sd_model.first_stage_model

        self.unet = UNetWrapper(sd_model.model, use_attn=False)
        
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

        


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x =  self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, x):
        # Use torch.no_grad() to prevent gradient computation on application of VQ encoder since it is frozen
        # Refer to paper for more info
        with torch.no_grad():
            # convert the input image to latent space and scale.
            latents = self.encoder_vq.encode(x).mode().detach() * self.config.model.params.scale_factor

        conditioning_scene_embedding = self.cide_module(x)

        t = torch.ones((x.shape[0],), device=x.device).long()

        outs = self.unet(latents, t, c_crossattn=[conditioning_scene_embedding])

        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x), outs[0]

class CIDE(nn.Module):
    def __init__(self, args, emb_dim):
        super().__init__()
        self.args = args
        self.vit_processor = ViTImageProcessor.from_pretrained('/root/hy58/caijin/huggingface_hub/google_vit')
        self.vit_model = ViTForImageClassification.from_pretrained('/root/hy58/caijin/huggingface_hub/google_vit')
        # self.vit_processor = ViTImageProcessor.from_pretrained(args.vit_model, resume_download=True)
        # self.vit_model = ViTForImageClassification.from_pretrained(args.vit_model, resume_download=True)
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(1000, 400),
            nn.GELU(),
            nn.Linear(400, args.no_of_classes)
        )
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
        
        self.embeddings = nn.Parameter(torch.randn(self.args.no_of_classes, self.dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)
        
    def pad_to_make_square(self, x):
        y = 255*((x+1)/2)
        y = torch.permute(y, (0,2,3,1))
        bs, _, h, w = x.shape
        if w>h:
            patch = torch.zeros(bs, w-h, w, 3).to(x.device)
            y = torch.cat([y, patch], axis=1)
        else:
            patch = torch.zeros(bs, h, h-w, 3).to(x.device)
            y = torch.cat([y, patch], axis=2)
        return y.to(torch.int)
    
    def forward(self, x):
        
        # make the image of dimension 480*640 into a square and downsample to utilize pretrained knowledge in the ViT
        y = self.pad_to_make_square(x)
        # use torch.no_grad() to prevent gradient flow through the ViT since it is kept frozen
        with torch.no_grad():
            inputs = self.vit_processor(images=y, return_tensors="pt").to(x.device)
            vit_outputs = self.vit_model(**inputs)
            vit_logits = vit_outputs.logits
            
        class_probs = self.fc(vit_logits)
        class_probs = self.m(class_probs)
        
        class_embeddings = class_probs @ self.embeddings
        conditioning_scene_embedding = self.embedding_adapter(class_embeddings, self.gamma) 
        
        return conditioning_scene_embedding
        
        
class EcoDepth3(nn.Module):
    def __init__(self, args=None, min_depth = 0.1):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.args = args
        embed_dim = 192
        channels_in = embed_dim*8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=channels_in, dataset='nyu', args = args)
        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
            
        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)



    def forward(self, x):  
        b, c, h, w = x.shape

        x = x*2.0 - 1.0  # normalize to [-1, 1]

        conv_feats, context = self.encoder(x)
        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]

        out = self.decoder([conv_feats]) #2,192,512,640

        out_depth = self.last_layer_depth(out) #2,1,512,640
           
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': [out_depth]}

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels
        self.args = args    
        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )
    
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        # import ipdb;ipdb.set_trace()
        out = self.deconv_layers(conv_feats[0])
        # debug_mode('shape', out)

        out = self.conv_layers(out) ###2 192 128 160
        # debug_mode('shape', out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)


class AutoRegDecoder(nn.Module):
    def __init__(self, min_depth=0.1, max_depth=10, hidden_dim=128, q_dim=128, memory_dim=1536, context_dim=192, patch_hws_list=None, channels_in=128*8, channels_out=128, args=None):
        super().__init__()

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hidden_dim = hidden_dim
        self.q_dim = q_dim
        self.context_dim = context_dim
        self.patch_hws_list = patch_hws_list


        # self.update = BasicUpdateBlockDepth(hidden_dim=hidden_dim, q_dim=q_dim,
        # context_dim=context_dim, patch_hws_list=patch_hws_list)

        self.update = BasicUpdateBlockDepth(hidden_dim=hidden_dim, q_dim=q_dim, memory_dim=memory_dim,
        context_dim=context_dim, patch_hws_list=patch_hws_list, channels_in=channels_in, 
        channels_out=channels_out, args=args)

        self.depth_num = 16

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, conv_feats, context, cond_feats=None):
    # def forward(self, conv_feats, cond_feats, context):

        x = conv_feats[0]
        b, c, h, w = context.shape
        device = context.device
        depth = torch.zeros([b, 1, h, w]).to(device)
    
    
        max_tree_depth = len(self.patch_hws_list)
        double_time = False
        
        if cond_feats != None:
            pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = self.update(depth, context, x, max_tree_depth, self.depth_num, self.min_depth, self.max_depth, self.patch_hws_list, cond_feats)
        
        else:
            pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = self.update(depth, context, x, max_tree_depth, self.depth_num, self.min_depth, self.max_depth, self.patch_hws_list)

        for i in range(len(pred_depths_r_list)):
            pred_depths_r_list[i] = self.upsample(pred_depths_r_list[i])
        for i in range(len(pred_depths_c_list)):
            pred_depths_c_list[i] = self.upsample(pred_depths_c_list[i]) 
        for i in range(len(uncertainty_maps_list)):
            uncertainty_maps_list[i] = self.upsample(uncertainty_maps_list[i]) 

        return pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list


class EcoDepth2(nn.Module):
    def __init__(self, args=None, min_depth = 0.1):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.args = args
        embed_dim = 192
        channels_in = embed_dim*8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=channels_in, dataset='nyu', args = args)
        # pretrained = '/ossfs/workspace/ts/wangjinhong/code/DepthAR/swin_large_patch4_window12_384_22k.pth'
        # self.encoder = NewCrfEncoder(pretrained=pretrained, out_dim=channels_in)
        hidden_dim=192
        q_dim=128
        context_dim=320
        # patch_hws_list=[(30,40), (30,40), (60, 80), (60, 80), (120, 160)]
        patch_hws_list=[(16,20), (32,40), (64, 80), (128, 160)]
        # patch_hws_list=[(32,40), (64, 80), (128, 160)]

        self.decoder = AutoRegDecoder(self.min_depth, self.max_depth, hidden_dim, q_dim, context_dim, patch_hws_list)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]

        conv_feats, context = self.encoder(x)

        # debug_mode('shape', context)
        context = F.interpolate(context, size=(120, 160), mode='bilinear') # 4,320,120,160
        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]  # shape:[4, 1536, 16, 20]

        pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = self.decoder([conv_feats], context)

        return {'pred_d': pred_depths_r_list}


class EcoDepth4(nn.Module):
    def __init__(self, args=None, min_depth = 0.1):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.args = args
        embed_dim = 192
        channels_in = embed_dim*8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=channels_in, dataset='nyu', args = args)
        # pretrained = '/ossfs/workspace/ts/wangjinhong/code/DepthAR/swin_large_patch4_window12_384_22k.pth'
        # self.encoder = NewCrfEncoder(pretrained=pretrained, out_dim=channels_in)
        hidden_dim=16 #depth_prompt
        q_dim=512 #transformer dim
        context_dim=1536 #feature map
        memory_dim=1536

        # patch_hws_list=[(16,20), (32,40), (64, 80), (128, 160)]
        patch_hws_list=[(4,5), (8,10), (16, 20), (32, 40)]

        self.decoder = AutoRegDecoder(self.min_depth, self.max_depth, hidden_dim, q_dim, memory_dim, context_dim, patch_hws_list, channels_in, channels_out, args)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]

        conv_feats, context = self.encoder(x)

        # debug_mode('shape', context)
        context = F.interpolate(context, size=(120, 160), mode='bilinear') # 4,320,120,160
        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]  # shape:[4, 1536, 16, 20]

        pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = self.decoder([conv_feats], context)

        return {'pred_d': pred_depths_r_list}




class EcoDepth(nn.Module): # _iebins
    def __init__(self, args=None, min_depth = 0.1):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.args = args
        embed_dim = 128
        channels_in = embed_dim*8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=1536, dataset='nyu', args = args)
        # pretrained = '/ossfs/workspace/ts/wangjinhong/code/DepthAR/swin_large_patch4_window12_384_22k.pth'
        # self.encoder = NewCrfEncoder(pretrained=pretrained, out_dim=channels_in)
        hidden_dim=128 #depth_prompt
        q_dim=1024 #transformer dim
        context_dim=320 #feature map
        memory_dim=1536
        # patch_hws_list=[(16,20), (32,40), (64, 80), (128, 160)]
        patch_hws_list=[(4,5),(8,10),(12,15),(16,20)]
        self.decoder = AutoRegDecoder(self.min_depth, self.max_depth, hidden_dim, q_dim, memory_dim, context_dim, patch_hws_list, channels_in, channels_out, args)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]

        conv_feats, context = self.encoder(x)

        # debug_mode('shape', context)
        context = F.interpolate(context, size=(128, 160), mode='bilinear') # 4,320,120,160
        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]  # shape:[4, 1536, 16, 20]

        pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = self.decoder([conv_feats], context)

        return {'pred_d': pred_depths_r_list}




class EcoDepth_DAR_A(nn.Module):  #DAR_A
    def __init__(self, args=None, min_depth = 0.1):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.args = args
        embed_dim = 128
        channels_in = embed_dim*8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=1536, dataset='nyu', args = args)
        pretrained = '/ossfs/workspace/ts/wangjinhong/code/DepthAR/swin_large_patch4_window12_384_22k.pth'
        self.encoder2 = SwinEncoder(pretrained=pretrained)
        hidden_dim=128 #depth_prompt
        q_dim=1024 #transformer dim
        context_dim=320 #feature map
        memory_dim=1536
        # patch_hws_list=[(16,20), (32,40), (64, 80), (128, 160)]
        patch_hws_list=[(4,6),(6,8),(8,10),(12,16),(16,20)]
        self.patch_hws_list = patch_hws_list
        self.decoder = AutoRegDecoder(self.min_depth, self.max_depth, hidden_dim, q_dim, memory_dim, context_dim, patch_hws_list, channels_in, channels_out, args)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]

        conv_feats, context = self.encoder(x)
        cond_feats = []
        for i in range(5):
            ph, pw = self.patch_hws_list[i]
            x_n = F.interpolate(x, size=(ph*16, pw*16), mode='bilinear')
            feats = self.encoder2(x_n)
            cond_feats.append(feats)
            # debug_mode('shape{}'.format(i), feats)

        # debug_mode('shape', context)
        context = F.interpolate(context, size=(128, 160), mode='bilinear') # 4,320,128,160
        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]  # shape:[4, 1536, 16, 20]

        pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = self.decoder([conv_feats], context, cond_feats)

        return {'pred_d': pred_depths_r_list}






class EcoDepth_easy(nn.Module):  #easy
    def __init__(self, args=None, min_depth = 0.1):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.args = args
        embed_dim = 192
        channels_in = embed_dim*8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=channels_in, dataset='nyu', args = args)
        pretrained = '/ossfs/workspace/ts/wangjinhong/code/DepthAR/swin_large_patch4_window12_384_22k.pth'
        self.encoder2 = SwinEncoder(pretrained=pretrained)

        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()
        
        self.qformer = TransformerDecoder(dim=channels_in, mid_dim=channels_in*2, memory_dim=1024, num_heads=8, num_layers=7, use_long_skip=True)
        
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
            
        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
    def forward(self, x):
        b, c, h, w = x.shape

        feats = self.encoder2(x)

        x = x*2.0 - 1.0  # normalize to [-1, 1]

        conv_feats, context = self.encoder(x)

        # print(conv_feats.shape, feats.shape)

        b, _, ch, cw = conv_feats.shape
        conv_feats = torch.flatten(conv_feats, 2).transpose(1, 2)
        feats = torch.flatten(feats, 2).transpose(1, 2)
        out = self.qformer(conv_feats, feats, tgt_mask=None)

        out = out.transpose(1,2).view(b, -1, ch, cw)

        out = self.decoder([out]) #2,192,512,640

        out_depth = self.last_layer_depth(out) #2,1,512,640
           
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': [out_depth]}

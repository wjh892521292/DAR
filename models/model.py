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
from modules.models import UNetWrapper, EmbeddingAdapter, AutoregressiveDepth

from modules.xformer_decoder import TransformerDecoder, TransformerDecoder2
from modules.blocks import FeatureFusionBlock, _make_scratch

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




class DepthAnythingEncoder(nn.Module):
    def __init__(self, features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super().__init__()
        self.use_clstoken = use_clstoken

        self.backbone = torch.hub.load('/ossfs/workspace/ts/wangjinhong/code/DepthAR/dar_final/networks/dinov2_main/', 'dinov2_vitl14', source='local', pretrained=False)
        in_channels = self.backbone.blocks[0].attn.qkv.in_features
        

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))


               
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])


        use_bn = False
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = self._make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = self._make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = self._make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = self._make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 12
        
     
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(features, features, 3, stride=2, padding=1),
            nn.GroupNorm(16, features),
            nn.ReLU(),
            nn.Conv2d(features, features, 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(features, features, 3, stride=2, padding=1),
        )

        out_dim = 1536

        self.out_layer = nn.Sequential(
            nn.Conv2d(features * 4, out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )

        self.init_weights()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if pretrained == None: 
            pretrained = '/ossfs/workspace/ts/wangjinhong/code/Ecodepth/checkpoints/depth_anything_vitl14.pth'
        print(f'== Load encoder backbone from: {pretrained}')
     
        pretrained_net_dict = torch.load(pretrained)

        pretrained_net_dict = {k[11:]:v for k, v in pretrained_net_dict.items() if k[11:] in self.backbone.state_dict()}
    
        self.backbone.load_state_dict(pretrained_net_dict)

    def _make_fusion_block(self, features, use_bn, size = None):
        return FeatureFusionBlock(
            features,
            nn.ReLU(False),
            deconv=False,
            bn=use_bn,
            expand=False,
            align_corners=True,
            size=size,
        )

    def forward(self, imgs):
        h, w = imgs.shape[-2:]

        # with torch.no_grad():

        out_features = self.backbone.get_intermediate_layers(imgs, 4, return_class_token=True)

        patch_h, patch_w = h // 14, w // 14

        outs = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            outs.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = outs
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_4_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_3_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_2_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, size=layer_1_rn.shape[2:])
        
        outs = [path_1, path_2, path_3, path_4]
        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        
        return self.out_layer(x), layer_1_rn

        # out = self.scratch.output_conv1(path_1)
        # out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # out = self.scratch.output_conv2(out)
        
        # print('22', path_1.shape, path_2.shape, path_3.shape, path_4.shape, layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape)
        # return out, out


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
        xx = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(xx), outs[0]

class CIDE(nn.Module):
    def __init__(self, args, emb_dim):
        super().__init__()
        self.args = args
        self.vit_processor = ViTImageProcessor.from_pretrained('/ossfs/workspace/hy58/caijin/huggingface_hub/google_vit')
        self.vit_model = ViTForImageClassification.from_pretrained('/ossfs/workspace/hy58/caijin/huggingface_hub/google_vit')
        # self.vit_processor = ViTImageProcessor.from_pretrained('/.../huggingface_hub/google_vit')
        # self.vit_model = ViTForImageClassification.from_pretrained('/.../huggingface_hub/google_vit')
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

        out_b = self.conv_layers(out) ###2 192 128 160
        # debug_mode('shape', out)

        out = self.up(out_b)
        out = self.up(out)

        return out_b

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


    

        self.update = AutoregressiveDepth(hidden_dim=hidden_dim, q_dim=q_dim, memory_dim=memory_dim,
        context_dim=context_dim, patch_hws_list=patch_hws_list, channels_in=channels_in, 
        channels_out=channels_out, args=args)

        self.depth_num = 16

        if args.dataset == 'kitti':
            self.upsample = nn.Upsample(size=(392, 672), mode='bilinear', align_corners=False)
        else:
            if args.encoder == 'VIT-14':
                self.upsample = nn.Upsample(size=(504, 672), mode='bilinear', align_corners=False)
            elif args.encoder == 'VIT-16':
                self.upsample = nn.Upsample(size=(512, 640), mode='bilinear', align_corners=False)

    def forward(self, conv_feats, context, cond_feats=None):
    # def forward(self, conv_feats, cond_feats, context):

        x = conv_feats[0]
        b, c, h, w = context.shape
        device = context.device
        depth = torch.zeros([b, 1, h, w]).to(device)
    
    
        max_tree_depth = len(self.patch_hws_list)
        
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





class DAR(nn.Module):
    def __init__(self, args=None, min_depth = 0.1):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.args = args
        embed_dim = 192
        channels_in = embed_dim*8
        channels_out = embed_dim


        if args.encoder == 'VIT-14':
            self.encoder = DepthAnythingEncoder(features=320, out_channels=[256, 512, 1024, 1024], use_clstoken=False)
            # patch_hws_list=[(4,5), (9,12), (18,24), (36,48)]
            patch_hws_list=[(4,5),(8,10),(12,15),(16,20)]
            # patch_hws_list=[(4,5), (8,10), (16,20), (32,40)]

        elif args.encoder == 'VIT-16':
            self.encoder = EcoDepthEncoder(out_dim=1536, dataset='nyu', args = args)
            # patch_hws_list=[(4,5), (8,10), (16,20), (32,40), (64,80)]
            patch_hws_list=[(4,5),(8,10),(12,15),(16,20)]
        self.depth_num = 16
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.patch_hws_list = patch_hws_list
        hidden_dim=128 #depth_prompt
        q_dim=1536 #transformer dim
        context_dim=320 #feature map
        memory_dim=1536
        # patch_hws_list=[(16,20), (32,40), (64, 80), (128, 160)]
        # patch_hws_list=[(3,4), (6,8), (12,16), (24,32)]

        self.decoder = AutoRegDecoder(self.min_depth, self.max_depth, hidden_dim, q_dim, memory_dim, context_dim, patch_hws_list, channels_in, channels_out, args)

        
    def forward(self, x):  
        b, c, h, w = x.shape
        
        x = x*2.0 - 1.0  # normalize to [-1, 1]

        conv_feats, context = self.encoder(x)

        # context = F.interpolate(context, size=(conv_feats.shape[2]*2, conv_feats.shape[3]*2), mode='bilinear') # 4,320,120,160
        context = F.interpolate(context, size=(self.patch_hws_list[-1][0]*2, self.patch_hws_list[-1][1]*2), mode='bilinear') # 4,320,120,160
        # context = F.interpolate(context, size=(128, 160), mode='bilinear') # 4,320,120,160

        pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = self.decoder([conv_feats], context)

        return {'pred_d': pred_depths_r_list}



from numpy import ones_like
from omegaconf import OmegaConf

import torch as th
import torch
import math
import abc

from torch import nn, einsum

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from transformers import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextConfig, CLIPTextModel, CLIPTextTransformer, _expand_mask
from inspect import isfunction
from .xformer_decoder import TransformerDecoder, TransformerDecoder2
import torch.nn.functional as F
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None):
            h = self.heads

            q = self.to_q(x)
            is_cross = context is not None
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            
            attn2 = rearrange(attn, '(b h) k c -> h b k c', h=h).mean(0)
            controller(attn2, is_cross, place_in_unet)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.diffusion_model.named_children()

    for net in sub_nets:
        if "input_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "middle_block" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.max_size) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item for item in self.step_store[key]] for key in self.step_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, base_size=64, max_size=None):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.base_size = base_size
        if max_size is None:
            self.max_size = self.base_size // 2
        else:
            self.max_size = max_size

def register_hier_output(model):
    self = model.diffusion_model
    from ldm.modules.diffusionmodules.util import checkpoint, timestep_embedding
    def forward(x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            # import pdb; pdb.set_trace()
            h = module(h, emb, context)  ## suraj: error happening inside kitti at this line
            hs.append(h)
        h = self.middle_block(h, emb, context)
        out_list = []

        for i_out, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if i_out in [1, 4, 7]:
                out_list.append(h)
        h = h.type(x.dtype)

        out_list.append(h)
        return out_list
    
    self.forward = forward

class UNetWrapper(nn.Module):
    def __init__(self, unet, use_attn=True, base_size=512, max_attn_size=None, attn_selector='up_cross+down_cross') -> None:
        super().__init__()
        self.unet = unet
        self.attention_store = AttentionStore(base_size=base_size // 8, max_size=max_attn_size)
        self.size16 = base_size // 32
        self.size32 = base_size // 16
        self.size64 = base_size // 8
        self.use_attn = use_attn
        
        if self.use_attn:
            register_attention_control(unet, self.attention_store)
        register_hier_output(unet)#this fun changes the self.forward of unet
        self.attn_selector = attn_selector.split('+')

    def forward(self, *args, **kwargs):
        if self.use_attn:
            self.attention_store.reset()
        out_list = self.unet(*args, **kwargs)
        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn16, attn32, attn64 = self.process_attn(avg_attn) # in nyu, error is happening inside this fun
            out_list[1] = torch.cat([out_list[1], attn16], dim=1)
            out_list[2] = torch.cat([out_list[2], attn32], dim=1)
            if attn64 is not None:
                out_list[3] = torch.cat([out_list[3], attn64], dim=1)
        return out_list[::-1]


    def process_attn(self, avg_attn):   
        attns = {self.size16: [], self.size32: [], self.size64: []}
        for k in self.attn_selector:   # self.attn_selector = ['up_cross', 'down_cross']
            for up_attn in avg_attn[k]:
                size = int(math.sqrt(up_attn.shape[1]))
                # exactly at below line, error is happening in nyu with use_attn=True
                attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', h=size))
        attn16 = torch.stack(attns[self.size16]).mean(0)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        if len(attns[self.size64]) > 0:
            attn64 = torch.stack(attns[self.size64]).mean(0)
        else:
            attn64 = None
        return attn16, attn32, attn64

class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77, pool=True):
        super().__init__()

        version = "/ossfs/workspace/hy58/caijin/huggingface_hub/openai_clip/"
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

        self.pool = pool

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        if self.pool:
            z = outputs.pooler_output
        else:
            z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class AutoregressiveDepth(nn.Module):
    def __init__(self, hidden_dim=128, q_dim=128, memory_dim=1536, context_dim=320, patch_hws_list=None, channels_in=128*8, channels_out=128, args=None):
        super(AutoregressiveDepth, self).__init__()
        
        self.q_dim = q_dim
        self.encoder = ProjectionInputDepth(hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.p_head = PHead(channels_out, channels_out)
        self.transdecoder = TransformerDecoder(dim=q_dim, mid_dim=2048, memory_dim=memory_dim, num_heads=8, num_layers=9, use_long_skip=True)

        self.transdecoder.init_weights()

        self.q_token_nums = 20
        self.query_tokens = nn.Parameter(torch.zeros(1, self.q_token_nums, q_dim))

        if patch_hws_list != None:
            self.patch_hws_list = patch_hws_list


        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
            
        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

        self.gru = SepConvGRU(hidden_dim=channels_out, input_dim=self.encoder.out_chs+context_dim)


    def mask_generate(self, patch_hws_list):
        L = sum(ph*pw for (ph,pw) in patch_hws_list)
        # print(L, patch_hws_list)
        if L != 1:
            d: torch.Tensor = torch.cat([torch.full((ph*pw,), i) for i, (ph,pw) in enumerate(patch_hws_list)]).view(1, L, 1)
        else:
            d: torch.Tensor = torch.zeros(1, 1).view(1, 1, 1)
            
        dT = d.transpose(1, 2)

        lvl_1L = dT[:, 0].contiguous()
        # attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(L, L)

        return attn_bias_for_masking


    def forward(self, depth, context, input_hidden, seq_len, depth_num, min_depth, max_depth, patch_list):
 
        pred_depths_r_list = []
        pred_depths_c_list = []
        uncertainty_maps_list = []

        b, _, h, w = depth.shape
        depth_range = max_depth - min_depth
        interval = depth_range / depth_num
        interval = interval * torch.ones_like(depth)
        interval = interval.repeat(1, depth_num, 1, 1)
        interval = torch.cat([torch.ones_like(depth) * min_depth, interval], 1)

        bin_edges = torch.cumsum(interval, 1)
        current_depths = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        index_iter = 0
        
    
        # print(input_hidden.shape)
        memory = torch.flatten(input_hidden, 2).transpose(1, 2)  #1536, 16, 20

        # print(memory.shape)

        next_tokens = self.query_tokens.repeat(b, 1, 1)

       
        token_seq = next_tokens

        for i in range(0, seq_len):
            
            if i != 0:
                next_maps = F.interpolate(next_maps, size=patch_list[i], mode='bilinear') #b,1536,h,w
                next_tokens = torch.flatten(next_maps, 2).transpose(1, 2)

                token_seq = torch.cat((token_seq, next_tokens), dim=1)

            tgt_mask = self.mask_generate(patch_list[:i+1]).cuda()
      


            ph, pw = patch_list[i]

        

            output = self.transdecoder(token_seq, memory, tgt_mask=tgt_mask)
            output = output[:, (-ph*pw):, :]
          
            output_maps = output[:, :, :].transpose(1, 2).view(b, -1, ph, pw) # b, 1024, ph, pw
            next_maps = output_maps


            output_maps = self.decoder(output_maps) #b, 128, 8*ph, 8*pw

            output_hidden = F.interpolate(output_maps, size=(h, w), mode='bilinear')
            input_features = self.encoder(current_depths.detach())
            input_c = torch.cat([input_features, context], dim=1)

            output_hidden = self.gru(output_hidden, input_c)

            pred_prob = self.p_head(output_hidden)
            depth_r = (pred_prob * current_depths.detach()).sum(1, keepdim=True)

            pred_depths_r_list.append(depth_r)

            uncertainty_map = torch.sqrt((pred_prob * ((current_depths.detach() - depth_r.repeat(1, depth_num, 1, 1))**2)).sum(1, keepdim=True))
            uncertainty_maps_list.append(uncertainty_map)
        
            index_iter = index_iter + 1

            pred_label = get_label(torch.squeeze(depth_r, 1), bin_edges, depth_num).unsqueeze(1)
            depth_c = torch.gather(current_depths.detach(), 1, pred_label.detach())
            pred_depths_c_list.append(depth_c)

            label_target_bin_left = pred_label

            left_min = torch.zeros_like(label_target_bin_left)
            # target_bin_left = torch.gather(bin_edges, 1, max(0, label_target_bin_left-1))
            target_bin_left = torch.gather(bin_edges, 1, torch.clamp_min(label_target_bin_left-2, left_min))

            label_target_bin_right = (pred_label.float() + 1).long()

            right_max = torch.ones_like(label_target_bin_right) * (depth_num-1)
            # target_bin_right = torch.gather(bin_edges, 1, min(15, label_target_bin_right+1))
            target_bin_right = torch.gather(bin_edges, 1, torch.clamp_max(label_target_bin_right+2, right_max))
            
            
            depth_start_update = torch.clamp_min(target_bin_left, min_depth)
            depth_range = (target_bin_right - target_bin_left).abs()

            interval = depth_range / depth_num
            interval = interval.repeat(1, depth_num, 1, 1)
            interval = torch.cat([torch.ones([b, 1, h, w], device=bin_edges.device) * depth_start_update, interval], 1)

            bin_edges = torch.cumsum(interval, 1).clamp(min_depth, max_depth)
            curr_depth = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        return pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list

def get_label(gt_depth_img, bin_edges, depth_num):

    with torch.no_grad():
        gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device)
        for i in range(depth_num):
            bin_mask = torch.ge(gt_depth_img, bin_edges[:, i])
            bin_mask = torch.logical_and(bin_mask, 
                torch.lt(gt_depth_img, bin_edges[:, i + 1]))
            gt_label[bin_mask] = i
        
        return gt_label

class PHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(PHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
    
    def forward(self, x, mask=None):
        out = self.conv2(F.relu(self.conv1(x)))
        out = torch.softmax(out, 1)
        return out

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+192):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1))) 
        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class ProjectionInputDepth(nn.Module):
    def __init__(self, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs 
        self.convd1 = nn.Conv2d(16, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        # self.convd3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd3 = nn.Conv2d(hidden_dim, out_chs, 3, padding=1)
        
    def forward(self, depth):
        d = F.relu(self.convd1(depth))
        d = F.relu(self.convd2(d))
        d = F.relu(self.convd3(d))
        # d = F.relu(self.convd4(d))
                
        return d

class Projection(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, 3, padding=1)
        
    def forward(self, x):
        out = self.conv(x)
                
        return out


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
        out = self.deconv_layers(conv_feats)
        # debug_mode('shape', out)

        out = self.conv_layers(out) ###2 192 128 160
        # debug_mode('shape', out)

        # out = self.up(out)
        # out = self.up(out)

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



def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

def upsample1(x, scale_factor=2, mode="bilinear"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)



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
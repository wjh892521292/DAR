# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat



# xFormers imports
try:
    from xformers.ops import memory_efficient_attention, unbind
    print("xFormers is available")
    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available", ImportError)
    XFORMERS_AVAILABLE = False
XFORMERS_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings as used in MoCo-v3

    Returns positional embedding of shape [B, H, W, D]
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, 'b (h w) d -> b d h w', h=h, w=w)
    return pos_emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, temb_dim=None, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if temb_dim is not None:
            self.adaLN_modulation = nn.Linear(temb_dim, 2 * self.hidden_features)

    def forward(self, x, temb=None):
        x = self.fc1(x)
        x = self.act(x)
        
        # Shift and scale using time emb (see https://arxiv.org/abs/2301.11093)
        if hasattr(self, 'adaLN_modulation'):
            shift, scale = self.adaLN_modulation(F.silu(temb)).chunk(2, dim=-1)
            x = modulate(x, shift, scale)
        
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if XFORMERS_AVAILABLE:
            q, k, v = unbind(qkv, 2)

            if mask is not None:
                mask = mask.unsqueeze(0).repeat(B, self.num_heads, 1, 1)
            x = memory_efficient_attention(q, k, v, attn_bias=mask)
            x = x.reshape([B, N, C])

        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale


            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, dim_context=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim_context = dim_context or dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim_context, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, mask=None):
        B, N, C = x.shape
        _, M, _ = context.shape

        # print(B, N, C, M)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads)

        if XFORMERS_AVAILABLE:
            k, v = unbind(kv, 2)

          

            x = memory_efficient_attention(q, k, v, attn_bias=None)
            x = x.reshape([B, N, C])

        else:
            q = q.permute(0, 2, 1, 3)
            kv = kv.permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, dim, dim_context, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, temb_in_mlp=False, temb_after_norm=True, temb_gate=True):
        super().__init__()
        dim_context = dim_context or dim
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, dim_context=dim_context, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim_context)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, temb_dim=None, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2*dim, dim) if skip else None

    def forward(self, x, context, tgt_mask=None, skip_connection=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip_connection], dim=-1))

        # if tgt_mask != None:
        x = x + self.drop_path(self.self_attn(self.norm1(x), tgt_mask))
        x = x + self.drop_path(self.cross_attn(self.query_norm(x), self.context_norm(context)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class DecoderBlock2(nn.Module):
    def __init__(self, dim, dim_context, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, temb_in_mlp=False, temb_after_norm=True, temb_gate=True):
        super().__init__()
        dim_context = dim_context or dim
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.cross_attn = CrossAttention(dim, dim_context=dim_context, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim_context)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, temb_dim=None, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2*dim, dim) if skip else None

    def forward(self, x, context, tgt_mask=None, skip_connection=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip_connection], dim=-1))

        # if tgt_mask != None:
        x = x + self.drop_path(self.self_attn(self.norm1(x), tgt_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class TransformerDecoder(nn.Module):


    def __init__(self, dim, mid_dim, memory_dim, num_heads, num_layers, use_long_skip: bool = False, mid_mlp_ratio: int = 4,
        mid_qkv_bias: bool = True,
        mid_drop_rate: float = 0.0,
        mid_attn_drop_rate: float = 0.0,
        mid_drop_path_rate: float = 0.0):
        super().__init__()
  
        self.use_long_skip = use_long_skip
        if use_long_skip:
            assert num_layers % 2 == 1, 'num_layers must be odd when using long skip connection'
        
        dpr = [x.item() for x in torch.linspace(0, mid_drop_path_rate, num_layers)] # stochastic depth decay rule

        self.mid_block = nn.ModuleList([
            DecoderBlock(
                dim=mid_dim, dim_context=memory_dim, num_heads=num_heads, 
                mlp_ratio=mid_mlp_ratio, qkv_bias=mid_qkv_bias, drop=mid_drop_rate, 
                attn_drop=mid_attn_drop_rate, drop_path=dpr[i], 
                skip=i > num_layers//2 and use_long_skip
            )
            for i in range(num_layers)
        ])

        self.mid_proj_in = nn.Linear(dim, mid_dim)
        self.mid_proj_out = nn.Linear(mid_dim, dim)
        
    def forward(self, x, memory, tgt_mask=None):
       
        x = self.mid_proj_in(x)

        if not self.use_long_skip:
            for blk in self.mid_block:
                x = blk(x, memory, tgt_mask=tgt_mask)
        else:
            skip_connections = []
            num_skips = len(self.mid_block) // 2
            for blk in self.mid_block[:num_skips]:
                x = blk(x, memory, tgt_mask=tgt_mask)
                skip_connections.append(x)
            x = self.mid_block[num_skips](x, memory, tgt_mask=tgt_mask)
            for blk in self.mid_block[num_skips+1:]:
                x = blk(x, memory, tgt_mask=tgt_mask, skip_connection=skip_connections.pop())

        x = self.mid_proj_out(x)
        return x
        
    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Handle already zero-init gates
            if "adaLN_gate" in name:
                continue
            # Handle ResNet gates that were not initialized by diffusers
            if "conv2" in name:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
                
            # Linear
            elif isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.init_std)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.conv_in' in name or '.conv_out' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

class TransformerDecoder2(nn.Module):


    def __init__(self, dim, mid_dim, memory_dim, num_heads, num_layers, use_long_skip: bool = False, mid_mlp_ratio: int = 4,
        mid_qkv_bias: bool = True,
        mid_drop_rate: float = 0.0,
        mid_attn_drop_rate: float = 0.0,
        mid_drop_path_rate: float = 0.0):
        super().__init__()
  
        self.use_long_skip = use_long_skip
        if use_long_skip:
            assert num_layers % 2 == 1, 'num_layers must be odd when using long skip connection'
        
        dpr = [x.item() for x in torch.linspace(0, mid_drop_path_rate, num_layers)] # stochastic depth decay rule

        self.mid_block = nn.ModuleList([
            DecoderBlock2(
                dim=mid_dim, dim_context=memory_dim, num_heads=num_heads, 
                mlp_ratio=mid_mlp_ratio, qkv_bias=mid_qkv_bias, drop=mid_drop_rate, 
                attn_drop=mid_attn_drop_rate, drop_path=dpr[i], 
                skip=i > num_layers//2 and use_long_skip
            )
            for i in range(num_layers)
        ])

        self.mid_proj_in = nn.Linear(dim, mid_dim)
        self.mid_proj_out = nn.Linear(mid_dim, dim)
        
    def forward(self, x, memory, tgt_mask=None):
       
        x = self.mid_proj_in(x)

        if not self.use_long_skip:
            for blk in self.mid_block:
                x = blk(x, memory, tgt_mask=tgt_mask)
        else:
            skip_connections = []
            num_skips = len(self.mid_block) // 2
            for blk in self.mid_block[:num_skips]:
                x = blk(x, memory, tgt_mask=tgt_mask)
                skip_connections.append(x)
            x = self.mid_block[num_skips](x, memory, tgt_mask=tgt_mask)
            for blk in self.mid_block[num_skips+1:]:
                x = blk(x, memory, tgt_mask=tgt_mask, skip_connection=skip_connections.pop())

        x = self.mid_proj_out(x)
        return x

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Handle already zero-init gates
            if "adaLN_gate" in name:
                continue
            # Handle ResNet gates that were not initialized by diffusers
            if "conv2" in name:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
                
            # Linear
            elif isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.init_std)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.conv_in' in name or '.conv_out' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


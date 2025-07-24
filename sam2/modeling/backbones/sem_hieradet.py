import logging
from functools import partial
import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP

def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)
    return x

# class AdaptFormerAdapter(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         bottleneck: int = 384,
#         dropout: float = 0.1,
#         init_option: str = "lora",
#         adapter_scalar: Union[str, float] = "0.1",
#         adapter_layernorm_option: str = "none",
#     ):
#         super().__init__()
#         self.n_embd = d_model
#         self.down_size = bottleneck

#         self.adapter_layernorm_option = adapter_layernorm_option
#         self.adapter_layer_norm_before = None
        
#         if adapter_layernorm_option in ["in", "out"]:
#             self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

#         if adapter_scalar == "learnable_scalar":
#             self.scale = nn.Parameter(torch.ones(1))
#         else:
#             self.scale = float(adapter_scalar)

#         self.down_proj = nn.Linear(self.n_embd, self.down_size)
#         self.non_linear_func = nn.ReLU()
#         self.up_proj = nn.Linear(self.down_size, self.n_embd)
#         self.dropout = dropout

#         if init_option == "lora":
#             with torch.no_grad():
#                 nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#                 nn.init.zeros_(self.up_proj.weight)
#                 nn.init.zeros_(self.down_proj.bias)
#                 nn.init.zeros_(self.up_proj.bias)

#     def forward(self, x, add_residual=True, residual=None):
#         residual = x if residual is None else residual

#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)
#         x = self.down_proj(x)
#         x = self.non_linear_func(x)
#         x = nn.functional.dropout(x, p=self.dropout, training=self.training)
#         x = self.up_proj(x)
#         x = x * self.scale

#         if self.adapter_layernorm_option == 'out':
#             x = self.adapter_layer_norm_before(x)

#         if add_residual:
#             return x + residual
#         else:
#             return x



class AdaptFormerAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        bottleneck: int = 384,
        num_inner_layers_pre=1,
        num_inner_layers_post=3,
        adapter_type="mlp",  # "mlp" or "group"
        dropout: float = 0.1,
        init_option: str = "lora",
        adapter_scalar: Union[str, float] = "0.1",
        adapter_layernorm_option: str = "none",
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_type = adapter_type

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        self.adapter_type = adapter_type

        if adapter_type == "mlp":
            self.inner_mlp_pre = nn.Sequential(*[
                nn.Sequential(nn.Linear(self.down_size, self.down_size), nn.ReLU())
                for _ in range(num_inner_layers_pre)
            ])
            self.inner_mlp_post = nn.Sequential(*[
                nn.Sequential(nn.Linear(self.n_embd, self.n_embd), nn.ReLU())
                for _ in range(num_inner_layers_post)
            ])
        elif adapter_type == "group":
            # Group style: simple one-layer bottleneck (down-up) only
            self.inner_mlp_pre = nn.Identity()
            self.inner_mlp_post = nn.Identity()
        else:
            pass
            # raise ValueError(f"Unsupported adapter_type: {adapter_type}")

        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj.bias)
                # nn.init.zeros_(self.up_proj.weight)
                nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.bias)
                if adapter_type == "mlp":
                    for layer in self.inner_mlp_pre:
                        nn.init.kaiming_uniform_(layer[0].weight, a=math.sqrt(5))
                        nn.init.zeros_(layer[0].bias)
                    for layer in self.inner_mlp_post:
                        nn.init.kaiming_uniform_(layer[0].weight, a=math.sqrt(5))
                        nn.init.zeros_(layer[0].bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        x = self.down_proj(x)
        x = self.non_linear_func(x)
        
        if self.adapter_type == "mlp":
            x = self.inner_mlp_pre(x)
            
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.up_proj(x)
        
        if self.adapter_type == "mlp":
            x = self.inner_mlp_post(x)
            
        x = x * self.scale

        if self.adapter_layernorm_option == 'out':
            x = self.adapter_layer_norm_before(x)

        if add_residual:
            return x + residual
        else:
            return x



class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)

        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)

        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        adapter: nn.Module = None,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(dim, dim_out, num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer)

        self.adapter = adapter  # <-- 增加 Adapter 成员

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: torch.Size([1, 256, 256, 144])
        shortcut = x
        x = self.norm1(x)

        if self.dim != self.dim_out:
            # self.dim == self.dim_out == 144
            shortcut = do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)
        if self.q_stride:
            # self.q_stride: None
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        if self.window_size > 0:
            # self.window_size: 8
            x = window_unpartition(x, window_size, pad_hw, (H, W))
                
        # self.drop_path: Identity()
        x = shortcut + self.drop_path(x)
        
        if self.adapter is not None:
            with torch.cuda.amp.autocast(enabled=False):
                adapter_x = x.to(self.adapter.down_proj.weight.dtype)  # 转成 float32
                adapter_x = self.adapter(adapter_x, add_residual=False)
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
       
        if self.adapter is not None:
            x = x + adapter_x

        return x

class Hiera(nn.Module):
    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        weights_path=None,
        return_interm_layers=True,
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)

        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        cur_stage = 1
        self.blocks = nn.ModuleList()
        
        # 添加适配器的起始位置（最后两层）
        # adapter_start_idx = depth - 2  # e.g., total depth - 
        # adapter_start_idx = depth - 6  # e.g., total depth - 
        adapter_start_idx = depth - 24  # e.g., total depth - 
        # adapter_start_idx = 0  # e.g., total depth - 

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks and i in self.global_att_blocks:
                window_size = 0

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            use_adapter = i >= adapter_start_idx  # 只在最后两层插入

            adapter = None
            if use_adapter:
                adapter = AdaptFormerAdapter(
                    d_model=dim_out,
                    # bottleneck=64,
                    bottleneck=384,
                    dropout=0.1,
                    init_option="lora",
                    num_inner_layers_pre=1,
                    num_inner_layers_post=3,
                    # adapter_type="mlp",  # or "group"
                    adapter_type="none",
                    adapter_scalar="0.1",
                    adapter_layernorm_option="none",
                )

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                adapter=adapter,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers else [self.blocks[-1].dim_out]
        )

        # self.abcd = nn.LayerNorm([256, 256, 144], eps=1e-6)
        # self.abcd = nn.Linear(144, 144)
        
        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        return pos_embed.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        # x.requires_grad_()
        # x = self.abcd(x)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name):
        num_layers = self.get_num_layers()
        if "rel_pos" in layer_name:
            return num_layers + 1
        elif "pos_embed" in layer_name or "patch_embed" in layer_name:
            return 0
        elif "blocks" in layer_name:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)

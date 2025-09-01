import torch
from torch import nn

class ClipSemanticFusion(nn.Module):
    """
    Fuse a spatial feature map (maskmem_features) with a pooled CLIP token (pooled_clip_feature).
    Supports at least 'concat_conv' fusion (concatenate token map with spatial features and use a 1x1 conv).
    
    Inputs:
      - maskmem_features: Tensor of shape [B, C, H, W]
      - pooled_clip_feature: Tensor of shape [B, C_clip]  (e.g., 1536 for CLIP)
    
    Output:
      - fused features: Tensor of shape [B, C, H, W]  (same shape as maskmem_features)
    
    Notes:
      - The module will project pooled_clip_feature to C channels via a Linear layer, then
        expand spatially and concatenate along channel dimension before a 1x1 conv that
        reduces back to C channels.
      - This is a learnable fusion (more flexible than simple broadcasting addition).
    """
    def __init__(self, in_channels, clip_dim, mode="concat_conv", use_bn=False, activation=True):
        super().__init__()
        assert mode in ("concat_conv",), "Currently only 'concat_conv' is implemented."
        self.mode = mode
        self.in_channels = in_channels
        self.clip_dim = clip_dim
        self.use_bn = use_bn
        self.activation = activation
        
        # Project CLIP pooled token to the same channel dimension as the feature map
        self.clip_proj = nn.Linear(clip_dim, in_channels)
        
        # After concatenation we have 2*in_channels -> reduce back to in_channels
        self.fusion_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()
    
    def forward(self, maskmem_features: torch.Tensor, pooled_clip_feature: torch.Tensor):
        """
        maskmem_features: [B, C, H, W]
        pooled_clip_feature: [B, clip_dim]
        returns: [B, C, H, W]
        """
        B, C, H, W = maskmem_features.shape
        assert C == self.in_channels, f"maskmem feature channels ({C}) != in_channels ({self.in_channels})"
        assert pooled_clip_feature.shape[0] == B, "Batch size mismatch between features and pooled clip token"
        assert pooled_clip_feature.shape[1] == self.clip_dim, f"Expected pooled_clip_feature dim {self.clip_dim}"
        
        # Project and reshape clip token -> [B, C, 1, 1], then expand to spatial size
        clip_proj = self.clip_proj(pooled_clip_feature)   # [B, C]
        clip_map = clip_proj.view(B, C, 1, 1).expand(-1, -1, H, W)  # [B, C, H, W]
        
        # Concatenate and fuse
        x = torch.cat([maskmem_features, clip_map], dim=1)  # [B, 2C, H, W]
        x = self.fusion_conv(x)                            # [B, C, H, W]
        x = self.bn(x)
        x = self.act(x)
        return x
    
    
class ClipResidualFusion(nn.Module):
    """
    Residual fusion of spatial feature maps with pooled CLIP features.

    Args:
        in_channels: channel dim of maskmem_features
        clip_dim: dim of pooled CLIP token (e.g., 1536)
        proj_dim: intermediate dim for CLIP projection (default=in_channels)
        activation: "relu" | "gelu" | "silu"
    """
    def __init__(self, in_channels, clip_dim, proj_dim=None, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.clip_dim = clip_dim
        self.proj_dim = proj_dim if proj_dim is not None else in_channels

        # 1) Project CLIP token to proj_dim
        self.clip_proj = nn.Linear(clip_dim, self.proj_dim)

        # 2) Fusion block (residual delta)
        hidden_dim = max(in_channels, self.proj_dim)  # safeguard
        act_layer = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
        }[activation]

        # Conv -> LN (channel-wise) -> activation -> Conv
        self.fusion_conv1 = nn.Conv2d(in_channels + self.proj_dim, hidden_dim, kernel_size=1)
        self.ln = nn.LayerNorm(hidden_dim)  # channel-wise LN
        self.act = act_layer
        self.fusion_conv2 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, maskmem_features: torch.Tensor, pooled_clip_feature: torch.Tensor):
        """
        maskmem_features: [B, C, H, W]
        pooled_clip_feature: [B, clip_dim]
        returns: [B, C, H, W]
        """
        B, C, H, W = maskmem_features.shape
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"
        # 1) Project CLIP token and expand
        clip_proj = self.clip_proj(pooled_clip_feature)  # [B, proj_dim]
        clip_map = clip_proj.view(B, self.proj_dim, 1, 1).expand(-1, -1, H, W)  # [B, proj_dim, H, W]

        # 2) Concatenate
        fusion_input = torch.cat([maskmem_features, clip_map], dim=1)  # [B, C+proj_dim, H, W]

        # 3) First conv
        x = self.fusion_conv1(fusion_input)  # [B, hidden_dim, H, W]

        # 4) LN on channels: permute to [B, H, W, C], LN, permute back
        x = x.permute(0, 2, 3, 1)  # [B, H, W, hidden_dim]
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # [B, hidden_dim, H, W]

        # 5) Activation
        x = self.act(x)

        # 6) Second conv -> residual delta
        delta = self.fusion_conv2(x)  # [B, C, H, W]

        # 7) Residual connection
        out = maskmem_features + delta
        return out


# ---- quick test ----
if __name__ == "__main__":
    B, C, H, W = 2, 64, 64, 64
    clip_dim = 1536
    maskmem = torch.randn(B, C, H, W)
    pooled_clip = torch.randn(B, clip_dim)

    fusion = ClipResidualFusionV2(in_channels=C, clip_dim=clip_dim, proj_dim=256, activation="gelu")
    out = fusion(maskmem, pooled_clip)
    print("Output shape:", out.shape)  # should be [2, 64, 64, 64]
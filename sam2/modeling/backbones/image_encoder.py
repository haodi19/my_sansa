# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def visualize_token_pca_and_save_all(
    feature_map,
    orig_image_tensor,  # [1, 3, H, W]
    save_dir=".",
    basename="sample",
    mask=None,
    show=False
):
    """
    生成 token PCA 可视化图，保存原图、PCA图、拼接图。

    Args:
        feature_map (torch.Tensor): shape [1, C, H, W]
        orig_image_tensor (torch.Tensor): [1, 3, H, W] 原图 tensor，值在 [0, 1] 或 [0, 255]
        save_dir (str): 保存文件夹
        basename (str): 文件名前缀，如 "dog1"
        mask (torch.Tensor): 可选，形状 [H, W]，token 选择区域
        show (bool): 是否可视化显示图像
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import cv2
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    def denormalize_image(tensor_img, mean, std):
        """
        将 normalize 过的图像还原为 0~255 范围的 RGB 图（float tensor -> uint8 numpy）
        Args:
            tensor_img: [1, 3, H, W] or [3, H, W]，值在 normalize 后的范围
            mean, std: list of 3 float
        Returns:
            uint8 np.ndarray, shape [H, W, 3]
        """
        if tensor_img.dim() == 4:
            tensor_img = tensor_img.squeeze(0)  # [3, H, W]
        
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor_img.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor_img.device)
        
        img = tensor_img * std + mean  # 还原
        img = img.clamp(0, 1)  # 限制范围
        img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        return img

    # === 1. 保存原图 ===
    orig_img = denormalize_image(
        tensor_img=orig_image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    orig_path = os.path.join(save_dir, f"{basename}_orig.png")
    cv2.imwrite(orig_path, orig_img_bgr)

    H_img, W_img = orig_img.shape[:2]

    # === 2. 处理 feature map 进行 PCA 可视化 ===
    B, C, H, W = feature_map.shape
    assert B == 1
    fmap = feature_map.squeeze(0).permute(1, 2, 0).contiguous()  # [H, W, C]
    fmap_np = fmap.reshape(-1, C).cpu().numpy()  # [H*W, C]

    if mask is not None:
        mask = mask.squeeze()
        assert mask.shape == (H, W)
        fmap_np = fmap_np[mask.reshape(-1) > 0]

    pca = PCA(n_components=3)
    pca_feat = pca.fit_transform(fmap_np)  # [N, 3]
    pca_feat -= pca_feat.min(0)
    pca_feat /= (pca_feat.max(0) + 1e-5)

    if mask is None:
        rgb_map = pca_feat.reshape(H, W, 3)
    else:
        rgb_map = np.zeros((H * W, 3))
        rgb_map[mask.reshape(-1) > 0] = pca_feat
        rgb_map = rgb_map.reshape(H, W, 3)

    # resize 回原图大小
    rgb_map_up = cv2.resize(rgb_map, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    rgb_img = (rgb_map_up * 255).astype(np.uint8)
    pca_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    pca_path = os.path.join(save_dir, f"{basename}_pca.png")
    cv2.imwrite(pca_path, pca_img_bgr)

    # === 3. 拼接图像（原图 | PCA）===
    concat_img = np.concatenate([orig_img_bgr, pca_img_bgr], axis=1)
    concat_path = os.path.join(save_dir, f"{basename}_concat.png")
    cv2.imwrite(concat_path, concat_img)

    # === 4. 显示（可选）===
    if show:
        plt.imshow(cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Original | Token PCA")
        plt.show()

    print(f"✅ Saved to:\n - {orig_path}\n - {pca_path}\n - {concat_path}")

class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # Forward through 
        features, pos = self.neck(self.trunk(sample))
        # f2=self.trunk(sample)
        # import pdb
        # pdb.set_trace()
        # visualize_token_pca_and_save_all(
        #     feature_map=features[2].to(dtype=torch.float32),
        #     orig_image_tensor=sample,
        #     save_dir="./vis3",
        #     basename="dog1",
        #     show=False  # 可视化看看效果
        # )
        
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos

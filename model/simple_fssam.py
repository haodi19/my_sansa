
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.backbones.sem_hieradet import AdaptFormerAdapter
from sam2.utils.misc import load_video_frames_from_data
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor, CLIPVisionModel
import open_clip
# def weighted_dice_loss2(prediction, target_seg, weighted_val: float = 1.0, reduction: str = "sum", eps: float = 1e-8):
#     target_seg = (target_seg == 1).float()  # B, H, W
#     n, h, w = target_seg.shape
#     prediction = prediction.reshape(-1, h, w)  # B, H, W
#     target_seg = target_seg.reshape(-1, h, w)
#     prediction = torch.sigmoid(prediction)
#     prediction = prediction.reshape(-1, h * w)  # B, H*W
#     target_seg = target_seg.reshape(-1, h * w)
#     loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
#     loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
#     loss = loss * weighted_val
#     if reduction == "sum":
#         loss = loss.sum() / n
#     elif reduction == "mean":
#         loss = loss.mean()
#     return loss

def weighted_dice_loss(prediction, target_seg, weighted_val: float = 1.0, reduction: str = "sum",
                       eps: float = 1e-8, ignore_index: int = 255):
    # 创建 valid mask（不是 ignore_index 的地方才算）
    valid_mask = (target_seg != ignore_index)  # B, H, W

    # 只考虑前景 class == 1 的部分
    target_seg = (target_seg == 1).float()  # B, H, W
    valid_mask = valid_mask.float()

    n, h, w = target_seg.shape
    prediction = prediction.reshape(-1, h, w)
    target_seg = target_seg.reshape(-1, h, w)
    valid_mask = valid_mask.reshape(-1, h, w)

    prediction = torch.sigmoid(prediction)

    # 应用 valid mask
    prediction = prediction * valid_mask
    target_seg = target_seg * valid_mask

    prediction = prediction.reshape(-1, h * w)
    target_seg = target_seg.reshape(-1, h * w)
    valid_mask = valid_mask.reshape(-1, h * w)

    # 计算 Dice loss（只考虑 valid 区域）
    inter = (target_seg * prediction).sum(dim=-1)
    denom = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)

    # 修正 denominator：防止全为 ignore 导致 denom = 0
    loss = 1 - 2 * inter / torch.clamp(denom, min=eps)
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum() / n
    elif reduction == "mean":
        loss = loss.mean()

    return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weighted_val: float = 1.0, reduction: str = "sum", ignore_index: int = 255):
        super(WeightedDiceLoss, self).__init__()
        self.weighted_val = weighted_val
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, prediction, target_seg):
        return weighted_dice_loss(
            prediction, target_seg,
            weighted_val=self.weighted_val,
            reduction=self.reduction,
            ignore_index=self.ignore_index
        )
    
# def weighted_bce_loss2(
#     prediction,
#     target_seg,
#     weighted_val: float = 1.0,
#     reduction: str = "sum",
#     eps: float = 1e-8
# ):  
#     if prediction.dim() == 4:
#         prediction = prediction.squeeze(1)
#     if target_seg.dim() == 4:
#         target_seg = target_seg.squeeze(1)

#     target_seg = (target_seg == 1).float()
#     n, h, w = target_seg.shape
    
#     # prediction = torch.sigmoid(prediction)
#     prediction = prediction.reshape(-1, h * w)
#     target_seg = target_seg.reshape(-1, h * w)

#     bce = F.binary_cross_entropy_with_logits(prediction, target_seg, reduction="none")
#     loss = bce * weighted_val

#     if reduction == "sum":
#         loss = loss.mean(1).sum() / n
#     elif reduction == "mean":
#         loss = loss.mean()
#     return loss

def weighted_bce_loss(
    prediction,
    target_seg,
    weighted_val: float = 1.0,
    reduction: str = "sum",
    eps: float = 1e-8,
    ignore_index: int = 255
):  
    if prediction.dim() == 4:
        prediction = prediction.squeeze(1)
    if target_seg.dim() == 4:
        target_seg = target_seg.squeeze(1)

    # Create valid mask
    valid_mask = (target_seg != ignore_index).float()

    # Convert foreground target
    target_seg = (target_seg == 1).float()
    
    n, h, w = target_seg.shape
    prediction = prediction.reshape(-1, h * w)
    target_seg = target_seg.reshape(-1, h * w)
    valid_mask = valid_mask.reshape(-1, h * w)

    # Compute BCE
    bce = F.binary_cross_entropy_with_logits(prediction, target_seg, reduction="none")

    # Mask out ignore pixels
    bce = bce * valid_mask

    # Apply weighting
    loss = bce * weighted_val

    if reduction == "sum":
        # mean over valid pixels per sample, then sum over batch
        valid_pixel_count = valid_mask.sum(dim=1).clamp(min=eps)
        loss = loss.sum(dim=1) / valid_pixel_count  # mean per image
        loss = loss.sum() / n
    elif reduction == "mean":
        valid_pixel_count = valid_mask.sum().clamp(min=eps)
        loss = loss.sum() / valid_pixel_count

    return loss


class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, reduction: str = "sum", ignore_index: int = 255):
        super(CombinedBCEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, prediction, target_seg, return_components=False):
        dice = weighted_dice_loss(prediction, target_seg, weighted_val=1.0, reduction=self.reduction, ignore_index=self.ignore_index)
        bce = weighted_bce_loss(prediction, target_seg, weighted_val=1.0, reduction=self.reduction, ignore_index=self.ignore_index)
        total = self.dice_weight * dice + self.bce_weight * bce
        if return_components:
            return total, dice.detach(), bce.detach()
        else:
            return total

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTokenHead(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, downsample_factor=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.downsample_factor = downsample_factor  # 通常为16（从1024到64）

    def forward(self, feat_flat, mask):
        """
        feat_flat: [4096, bs, 256] (flattened spatial tokens)
        mask: [bs, H, W] (e.g., 1024 x 1024)

        return:
        semantic_token: [bs, output_dim]
        """
        B = mask.shape[0]
        H_feat = W_feat = int(feat_flat.shape[0] ** 0.5)  # should be 64
        D = feat_flat.shape[2]

        # 1. Downsample mask to 64x64
        mask_down = F.interpolate(mask.unsqueeze(1).float(), size=(H_feat, W_feat), mode='bilinear', align_corners=False)  # [B, 1, 64, 64]
        mask_down = mask_down.squeeze(1)  # [B, 64, 64]
        mask_flat = mask_down.view(B, -1)  # [B, 4096]

        # 2. Apply mask to features
        feat_flat = feat_flat.permute(1, 0, 2)  # [B, 4096, D]
        masked_feat = feat_flat * mask_flat.unsqueeze(-1)  # [B, 4096, D]

        # 3. Normalize by valid area and pool
        eps = 1e-6
        valid_area = mask_flat.sum(dim=1, keepdim=True) + eps  # [B, 1]
        pooled = masked_feat.sum(dim=1) / valid_area  # [B, D]

        # 4. Projection
        sem_token = self.proj(pooled)  # [B, output_dim]

        return sem_token
    
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

# sam2_tmp = build_sam2_video_predictor(config_file='tmp.yaml', ckpt_path='/hdd0/ljn/new_sam2/my_fssam/pretrained/sam2.1_hiera_large.pt', mode=None)
# sam2_tmp = sam2_tmp.to(torch.bfloat16).cuda()

class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.dataset = args.data_set
        # self.criterion = WeightedDiceLoss()
        self.criterion = CombinedBCEDiceLoss(dice_weight=1.0, bce_weight=20.0, reduction="sum")
        self.print_freq = args.print_freq / 2
        self.pretrained = True
        self.classes = 1

        # Build SAM2
        self.sam2_weight = args.sam2_weight
        self.sam2_config = args.sam2_config
        self.sam2 = build_sam2_video_predictor(config_file=self.sam2_config, ckpt_path=self.sam2_weight, mode=None)
        
        self.use_sem_head = False
        if self.use_sem_head:
            self.sem_head = SemanticTokenHead(input_dim=256, output_dim=256)
        
        self.use_sem_visual_encoder = True
        if self.use_sem_visual_encoder:
            self.sem_visual_model, _, _ = open_clip.create_model_and_transforms("convnext_large_d_320", pretrained="laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin")
            self.clip_fusion = ClipSemanticFusion(in_channels=64, clip_dim=1536, mode="concat_conv", use_bn=True)
        # self.use_text_prompt = args.use_text_prompt
        self.use_text_prompt = False
        
        if self.use_text_prompt:       
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_model.eval()
            # self.text_model = self.text_model.to('cuda')
            
            in_dim = 512
            out_dim = 256
            self.text_fc = nn.Sequential(
                nn.Linear(in_dim, in_dim),   # 保持512维
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim),  # 映射到256维
                nn.Dropout(0.0)
            )

    def get_optim(self, model, args, LR, type = 'sam2'):
        if type == 'sam2':
            optimizer = torch.optim.AdamW(
                [
                    {'params': model.sam2.sam_mask_decoder.parameters()},
                    {'params': model.sam2.memory_encoder.parameters()},
                    {'params': model.sam2.memory_attention.parameters()},
                ], lr=LR, weight_decay=args.weight_decay
            )
        elif type == 'sansa':
            optimizer = torch.optim.AdamW(
                [
                    # {'params': model.sam2.image_encoder.trunk.blocks[46].adapter.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.blocks[47].adapter.parameters()},
                    # {'params':  model.sam2.image_encoder.trunk.blocks[45].adapter.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.blocks[44].adapter.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.blocks[43].adapter.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.blocks[42].adapter.parameters()},
                    
                    # {'params': model.sam2.image_encoder.trunk.blocks[47].mlp.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.blocks[46].mlp.parameters()},
                    # {'params': model.sam2.sam_mask_decoder.parameters()},
                    # {'params': model.sam2.memory_encoder.parameters()},
                    # {'params': model.sam2.memory_attention.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.blocks[3].mlp.layers[0].parameters()},
                    # {'params': model.sam2.image_encoder.trunk.abcd.parameters()},
                     {'params': model.sam2.image_encoder.trunk.blocks[i].adapter.parameters()} for i in range(24,47)  # 包括第0层到第46层
                ],
                lr=LR,
                weight_decay=args.weight_decay
            )
        elif type == 'sansa_text':
            optimizer = torch.optim.AdamW(
                [
                    {'params': model.sam2.image_encoder.trunk.blocks[46].adapter.parameters()},
                    {'params': model.sam2.image_encoder.trunk.blocks[47].adapter.parameters()},
                    {'params': model.sam2.image_encoder.trunk.blocks[45].adapter.parameters()},
                    {'params': model.sam2.image_encoder.trunk.blocks[44].adapter.parameters()},
                    {'params': model.sam2.image_encoder.trunk.blocks[43].adapter.parameters()},
                    {'params': model.sam2.image_encoder.trunk.blocks[42].adapter.parameters()},
                    {'params': model.text_fc.parameters()},
                    # {'params': model.sam2.sam_mask_decoder.parameters()},
                    # {'params': model.sam2.memory_encoder.parameters()},
                    # {'params': model.sam2.memory_attention.parameters()},
                ],
                lr=LR,
                weight_decay=args.weight_decay
            )
        return optimizer

            
    def freeze_modules(self, model, type = 'sam2'):
        if type == 'sam2':
            for param in model.sam2.image_encoder.parameters():
                param.requires_grad = False
            for param in model.sam2.sam_prompt_encoder.parameters():
                param.requires_grad = False
            for param in model.sam2.obj_ptr_proj.parameters():
                param.requires_grad = False
            for param in model.sam2.mask_downsample.parameters():
                param.requires_grad = False            
        elif type == 'sansa':
            # 全部参数先冻结
            for name, param in model.named_parameters():
                # if 'adapter' not in name:
                if 'adapter' not in name and 'clip_fusion' not in name:
                # if 'adapter' not in name and '46.mlp' not in name and "47.mlp" not in name:
                # if 'trunk' not in name:
                # if 'trunk.blocks.3.mlp.layers.0' not in name:
                # if 'abcd' not in name:
                    param.requires_grad = False
                    
            # for param in model.sam2.sam_mask_decoder.parameters():
            #     param.requires_grad = True
            # for param in model.sam2.memory_encoder.parameters():
            #     param.requires_grad = True
            # for param in model.sam2.memory_attention.parameters():
            #     param.requires_grad = True
        elif type == 'sansa_text':
            # 全部参数先冻结
            for name, param in model.named_parameters():
                if 'adapter' not in name and 'text_fc' not in name:
                    param.requires_grad = False

            # for param in model.sam2.sam_mask_decoder.parameters():
            #     param.requires_grad = True
                
        # # 单独解冻 Hiera trunk 中的 Adapter 参数
        # for name, module in model.sam2.image_encoder.trunk.named_modules():
        #     if isinstance(module, AdaptFormerAdapter):
        #         for param in module.parameters():
        #             param.requires_grad = True
        
    def encode_class_names(self, class_names) -> torch.Tensor:
        """        
        Args:
            class_names (List[str] or Tuple[str]): 例如 ['cat', 'dog', 'zebra']
        
        Returns:
            torch.Tensor: shape 为 [N, 512]，L2 normalized 特征
        """
        # 自动构造带提示词的输入文本
        prompts = [f"A photo of a {name}" for name in class_names]
        
        # 编码
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            text_features = outputs.last_hidden_state[:, 0, :]  # 取每个输入的 [CLS] token，shape [N, 512]
            text_features = F.normalize(text_features, p=2, dim=-1)  # L2 normalize
        
        return text_features  # shape [N, 512]
    
    def encode_with_clip(self, x: torch.Tensor) -> dict:
        """
        输入:
            x: [B,3,H,W]  已经过你自己的 transform (0.485/0.229 normalize, ResizeLongSideAndPad)
            vision_model: open_clip.create_model_and_transforms() 得到的 model
            img_size: CLIP 模型输入大小 (ConvNeXt 是 320)

        输出:
            dict:
                'cls':   [B, hidden_dim]  全局向量
                'dense': [B,C,Hf,Wf]      空间特征 (来自 trunk.norm_pre)
        """

        # 1. 还原成 0-1 图像
        mean_old = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std_old  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = x * std_old + mean_old   # 还原到 [0,1]

        # 3. 应用 CLIP 官方的 mean/std
        mean_new = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
        std_new  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
        x = (x - mean_new) / std_new

        # 4. 前向提取特征
        out = {}
        with torch.no_grad():
            trunk = self.sem_visual_model.visual.trunk
            feats = trunk.stem(x)
            out['stem'] = feats.contiguous()

            for i in range(4):
                feats = trunk.stages[i](feats)
                out[f'res{i+2}'] = feats.contiguous()

            feats = trunk.norm_pre(feats)
            out['clip_vis_dense'] = feats.contiguous()

        return out

    def mask_pooling(self, x, mask):
        """
        Args:
            x:    [B, C, H, W]
            mask: [B, H, W]   (单mask版本)
        Returns:
            mask_pooled_x: [B, C]
        """
        B, C, H, W = x.shape

        # 调整 mask 尺寸
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(
                mask.unsqueeze(1),  # [B,1,H,W]
                size=(H, W),
                mode="nearest"
            ).squeeze(1)  # [B,H,W]

        with torch.no_grad():
            mask = (mask > 0).to(x.dtype)  # [B,H,W]
            denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8  # [B,1]

        # einsum 聚合
        # x: [B,C,H,W], mask: [B,H,W] -> [B,C]
        pooled = torch.einsum("bchw,bhw->bc", x, mask / denorm)

        return pooled

    def visualize_mask_on_image(self, image_tensor, mask_tensor, save_path='output.png', alpha=0.5):
        """
        将 1x1x128x128 的 mask 可视化到 1x3x512x512 的图像上，并保存为图片。
        
        Args:
            image_tensor (torch.Tensor): 输入图像，形状为 [1, 3, 512, 512]
            mask_tensor (torch.Tensor): 输入 mask，形状为 [1, 1, 128, 128]
            save_path (str): 保存路径
            alpha (float): mask 的透明度，0 到 1
        """
        import torch
        import torchvision.transforms.functional as TF
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        # 去 batch 维度
        image = image_tensor.squeeze(0)  # [3, 512, 512]
        mask = mask_tensor.squeeze(0)    # [1, 128, 128]

        # 将 mask 上采样到 512x512
        mask_up = TF.resize(mask, size=[512, 512], interpolation=TF.InterpolationMode.NEAREST)  # [1, 512, 512]
        mask_up = mask_up.squeeze(0)  # [512, 512]

        # 转换为 numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()  # [512, 512, 3]
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        mask_np = mask_up.cpu().numpy()  # [512, 512]

        # 创建红色遮罩
        red_mask = np.zeros_like(image_np)
        red_mask[..., 0] = 255  # 红色通道

        # 叠加红色 mask
        mask_bool = mask_np > 0.5
        overlay = np.where(mask_bool[..., None], 
                        (alpha * red_mask + (1 - alpha) * image_np).astype(np.uint8), 
                        image_np)

        # 保存图片
        Image.fromarray(overlay).save(save_path)
        print(f"Saved visualization to {save_path}")
    
    @autocast()
    def forward(self, x, s_x, s_y, y_m, cat_idx=None, priors=None, class_name=None, multi_frame_training=False):
        if multi_frame_training:
            return  self.forward_multi_frame(
                s_x=s_x, s_y=s_y, x=x, y_m=y_m, cat_idx=cat_idx, class_name=class_name
            )
        # x:torch.Size([bs, 3, H, W])
        # s_x:torch.Size([bs, shot, 3, H, W])
        # s_y:torch.Size([bs, shot, H, W])
        # y_m:torch.Size([bs, H, W])
        b, _, h, w = x.size()  # b=1, 3, H, W
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_x = s_x.view(-1, 3, h, w)  # b*s, 3, 512, 512
            # remove padding (255)
            s_mask = s_y.clone().float()
            s_mask[s_mask == 255] = 0  # 1, shot, H, W

            # ========================================
            # SAM2 - video mode
            # ========================================
            x = load_video_frames_from_data(x, offload_video_to_cpu=False)  # b, 3, 512, 512
            s_x = load_video_frames_from_data(s_x, offload_video_to_cpu=False)  # b*s, 3, 512, 512
            
            # with torch.no_grad():
            # obtain query and support features
            _, _, qry_feats, qry_poss, qry_sizes = self.sam2.get_image_feature_batch(x)
            _, _, sup_feats, sup_poss, sup_sizes = self.sam2.get_image_feature_batch(s_x)
            
            # qry_feats/sup_feats: list,多尺度特征
            # qry_feats[0]: torch.Size([65536, 1, 32])
            # qry_feats[1]: torch.Size([16384, 1, 64])
            # qry_feats[2]: torch.Size([4096, 1, 256])
            # qry_sizes: [(256, 256), (128, 128), (64, 64)]
            
            if self.use_sem_head:
                sem_token = self.sem_head(sup_feats[-1], s_mask.view(-1, h, w))  # [bs *shot, 256]
                sem_token = sem_token.reshape(b, 1, -1)
                
            # add support prompt - gt mask
            sup_fg = s_mask[:, 0, ...].unsqueeze(1)  # b, 1, h, w
            (sup_fg_preds, sup_fg_obj_ptrs, sup_fg_mem_feats, sup_fg_mem_poss) = self.sam2.add_new_mask_batch(
                sup_feats, sup_sizes, sup_fg
            )  # support fg gt memory, b, 64, 32, 

            # import pdb
            # pdb.set_trace()
            # self.visualize_mask_on_image(s_x, sup_fg_preds, save_path='vis_result.png')

            text_features = None
            if self.use_text_prompt:
                text_features = self.encode_class_names(class_name)
                text_features = self.text_fc(text_features)  # 输出为 [bs, 256]
            
            if self.use_sem_visual_encoder:
                # target_size = self.sem_visual_model.visual.image_size  # 336
                # q_sem_encoder_x = F.interpolate(x, size=target_size, mode="bicubic", align_corners=False)    \     
                qry_out = self.encode_with_clip(x)
                sup_out = self.encode_with_clip(s_x)
                # torch.Size([1, 1536, 32, 32]), 32 = 1024 / 32
                qry_clip_vis_dense = qry_out['clip_vis_dense']
                sup_clip_vis_dense = sup_out['clip_vis_dense']
                # 下采样 mask 到特征图大小, torch.Size([bs*shot, 32, 32])
                sup_mask_for_pooling = F.interpolate(s_mask.view(-1, 1, h, w).float(), size=sup_clip_vis_dense.shape[-2:], mode="nearest").squeeze(1)
                # pooled_clip_feature: torch.Size([1, 1, 1536])
                sup_pooled_clip_feature = self.mask_pooling(sup_clip_vis_dense, sup_mask_for_pooling)
                # sup_fg_mem_feats: torch.Size([1, 64, 64, 64])
                sup_fg_mem_feats = self.clip_fusion(sup_fg_mem_feats, sup_pooled_clip_feature)
            # visualize_token_pca_and_save_all(feature_map=sup_fg_mem_feats.to(dtype=torch.float32), orig_image_tensor=s_x[0][0].unsqueeze(0),save_dir="./vis5",basename="dog1",show=False)
            
            # propagate prompted frames (直接用SAM2的propagate_in_video_batch)
            sup_mask = F.interpolate(s_mask[:, 0, ...].unsqueeze(1).float(), size=qry_sizes[-1], mode='nearest')
            low_res_masks, output_query, pix_feat_with_mem = self.sam2.propagate_in_video_batch_mine(
                qry_feats, qry_poss, qry_sizes,
                sup_fg_mem_feats, sup_fg_mem_poss, sup_fg_preds, sup_fg_obj_ptrs, text_features = text_features
            )
            output_query = output_query.squeeze(1)


            # Loss
            if self.training:
                # main_loss = self.criterion(output_query, y_m.float())
                main_loss, dice_loss_val, bce_loss_val = self.criterion(output_query, y_m.float(), return_components=True)
                aux_loss1 = torch.zeros_like(main_loss)
                aux_loss2 = torch.zeros_like(main_loss)
                return output_query, main_loss, aux_loss1, aux_loss2, dice_loss_val, bce_loss_val
            else:
                output_query = self.sam2.mask_refinement_batch(qry_feats, qry_sizes, low_res_masks, pix_feat_with_mem).squeeze(1)
                
                # _, _, qry_feats2, qry_poss2, qry_sizes2 = sam2_tmp.get_image_feature_batch(x)
                # output_query = sam2_tmp.mask_refinement_batch(qry_feats2, qry_sizes2, low_res_masks).squeeze(1)
            
                return output_query, None

    @autocast()
    def forward_multi_frame(self, x, s_x, s_y, y_m, cat_idx=None, priors=None, class_name=None):
        # x: torch.Size([bs, 3, 1024, 1024])
        # s_x: torch.Size([bs, shot, 3, 1024, 1024])
        # s_y: torch.Size([bs, shot, 1024, 1024])
        # y_m: torch.Size([bs, 1024, 1024])
        b, _, h, w = x.size()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # flatten support
            shot = s_x.size(1)
            full_seq = torch.cat([s_x[:, 1:], x.unsqueeze(1)], dim=1)  # shape: [bs, shot, 3, H, W]
            full_seq = full_seq.view(-1, 3, h, w)  # (bs*(shot)), 3, H, W
            ref_frame = s_x[:, 0]  # b, 3, h, w
            ref_mask = s_y[:, 0]  # b, h, w
            target_gt = torch.cat([s_y[:, 1:], y_m.unsqueeze(1)], dim=1)  # bs, shot, h, w

            ref_frame = load_video_frames_from_data(ref_frame, offload_video_to_cpu=False)  # b, 3, 512, 512
            full_seq = load_video_frames_from_data(full_seq, offload_video_to_cpu=False)  # b*s, 3, 512, 512

            # extract features
            _, _, ref_feat, ref_pos, ref_sizes = self.sam2.get_image_feature_batch(ref_frame)
            _, _, tgt_feat, tgt_pos, tgt_sizes = self.sam2.get_image_feature_batch(full_seq)

            # init memory from ref frame
            sup_fg = ref_mask.unsqueeze(1).float()  # b,1,h,w
            valid_mask = (sup_fg == 255)
            
            sup_fg = sup_fg.masked_fill(valid_mask, 0) # 1, shot, H, W

            sup_fg_preds, sup_fg_obj_ptrs, sup_fg_mem_feats, sup_fg_mem_poss = self.sam2.add_new_mask_batch(
                ref_feat, ref_sizes, sup_fg
            )
            # sup_fg_mem_feats: torch.Size([bs, 64, 64, 64])
            
            text_features = None
            if self.use_text_prompt:
                text_features = self.encode_class_names(class_name)
                text_features = self.text_fc(text_features)  # 输出为 [bs, 256]

            if self.use_sem_visual_encoder:
                # target_size = self.sem_visual_model.visual.image_size  # 336
                # q_sem_encoder_x = F.interpolate(x, size=target_size, mode="bicubic", align_corners=False)    \     
                # qry_out = self.encode_with_clip(x)
                sup_out = self.encode_with_clip(ref_frame)
                # torch.Size([1, 1536, 32, 32]), 32 = 1024 / 32
                # qry_clip_vis_dense = qry_out['clip_vis_dense']
                sup_clip_vis_dense = sup_out['clip_vis_dense']
                # 下采样 mask 到特征图大小, torch.Size([bs*shot, 32, 32])
                sup_mask_for_pooling = F.interpolate(sup_fg.view(-1, 1, h, w).float(), size=sup_clip_vis_dense.shape[-2:], mode="nearest").squeeze(1)
                # pooled_clip_feature: torch.Size([1, 1, 1536])
                sup_pooled_clip_feature = self.mask_pooling(sup_clip_vis_dense, sup_mask_for_pooling)
                # sup_fg_mem_feats: torch.Size([1, 64, 64, 64])
                sup_fg_mem_feats = self.clip_fusion(sup_fg_mem_feats, sup_pooled_clip_feature)

            # memory containers
            memory_bank = {
                0: {
                    "maskmem_features": sup_fg_mem_feats,
                    "maskmem_pos_enc": [sup_fg_mem_poss[-1]],
                    "pred_masks": sup_fg_preds,
                    "obj_ptr": sup_fg_obj_ptrs,
                }
            }

            # for losses
            all_preds = []
            losses = []
            dice_vals = []
            bce_vals = []

            for j in range(shot):  # loop over [s_x[1:], x]
                idx = j  # time index
                # slice features for current 
                qry_frame = full_seq[j::shot]
                qry_feat = [f[:,j::shot] for f in tgt_feat]
                qry_pos = [p[:,j::shot] for p in tgt_pos]
                qry_gt = target_gt[:, j]

                # gather memory entries
                mem_feats = torch.cat([v["maskmem_features"] for v in memory_bank.values()], dim=0)
                mem_pos = torch.cat([v["maskmem_pos_enc"][-1] for v in memory_bank.values()], dim=0)
                mem_preds = torch.cat([v["pred_masks"] for v in memory_bank.values()], dim=0)
                mem_ptrs = torch.cat([v["obj_ptr"] for v in memory_bank.values()], dim=0)

                # propagate
                low_res_mask, output_query, pix_feat_with_mem = self.sam2.propagate_in_video_batch_mine_multi_frame(
                    qry_feat, qry_pos, tgt_sizes,
                    mem_feats, [mem_pos], mem_preds, mem_ptrs,
                    text_features=None
                )
                output_query = output_query.squeeze(1)  # [bs, h, w]
                all_preds.append(output_query)

                # loss
                if self.training:
                    loss, dice, bce = self.criterion(output_query, qry_gt.float(), return_components=True)
                    losses.append(loss)
                    dice_vals.append(dice)
                    bce_vals.append(bce)

                sup_fg = output_query.unsqueeze(1)
                sup_fg = sup_fg.masked_fill(valid_mask, 0)
                # update memory using predicted mask
                # pred_mask_up = F.interpolate(output_query.unsqueeze(1), size=ref_sizes[-1], mode='nearest')
                # pred_mask_up = pred_mask_up.detach()
                fg_preds, obj_ptrs, mem_feats_new, mem_pos_new = self.sam2.add_new_mask_batch(
                    qry_feat, tgt_sizes, sup_fg
                )
                
                if self.use_sem_visual_encoder:
                    import pdb
                    pdb.set_trace()
                    # target_size = self.sem_visual_model.visual.image_size  # 336
                    # q_sem_encoder_x = F.interpolate(x, size=target_size, mode="bicubic", align_corners=False)    \     
                    # qry_out = self.encode_with_clip(x)
                    sup_out = self.encode_with_clip(qry_frame)
                    # torch.Size([1, 1536, 32, 32]), 32 = 1024 / 32
                    # qry_clip_vis_dense = qry_out['clip_vis_dense']
                    sup_clip_vis_dense = sup_out['clip_vis_dense']
                    # 下采样 mask 到特征图大小, torch.Size([bs*shot, 32, 32])
                    sup_mask_for_pooling = F.interpolate(sup_fg.view(-1, 1, h, w).float(), size=sup_clip_vis_dense.shape[-2:], mode="nearest").squeeze(1)
                    # pooled_clip_feature: torch.Size([1, 1, 1536])
                    sup_pooled_clip_feature = self.mask_pooling(sup_clip_vis_dense, sup_mask_for_pooling)
                    # sup_fg_mem_feats: torch.Size([1, 64, 64, 64])
                    mem_feats_new = self.clip_fusion(mem_feats_new, sup_pooled_clip_feature)

                memory_bank[idx + 1] = {
                    "maskmem_features": mem_feats_new,
                    "maskmem_pos_enc": [mem_pos_new[-1]],
                    "pred_masks": fg_preds,
                    "obj_ptr": obj_ptrs
                }

            if self.training:
                return all_preds, torch.stack(losses).mean(), torch.zeros_like(losses[0]), torch.zeros_like(losses[0]), \
                    torch.stack(dice_vals).mean(), torch.stack(bce_vals).mean()
            else:
                return all_preds, None


def visualize_fewshot_seg(support_image, support_mask, query_image, query_mask, save_path='vis_fewshot.png',
                           support_overlay_path='output/support_overlay.png',
                           query_overlay_path='output/query_overlay.png',
                           support_img_path='output/support_img.png',
                           query_img_path='output/query_img.png'):
    """
    Visualize few-shot segmentation results with red-highlighted masks.

    Args:
        support_image (Tensor): [1, 3, H, W], 0–255 float or uint8
        support_mask (Tensor): [1, 1, H, W], binary/int/float
        query_image (Tensor): [1, 3, H, W], 0–255 float or uint8
        query_mask (Tensor): [1, H, W], binary/int/float
        save_path (str): File path to save the visualization
    """
    import torch
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    import os
    
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
    
    # Squeeze and move to CPU
    support_image = support_image[0].cpu()
    support_mask = support_mask[0][0].cpu().float()
    query_image = query_image[0].cpu()
    query_mask = query_mask[0].cpu().float()
    
    support_image = denormalize_image(
        tensor_img=support_image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    query_image = denormalize_image(
        tensor_img=query_image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Normalize image if needed
    if support_image.dtype == torch.float32 and support_image.max() > 1:
        support_image = support_image / 255.0
    if query_image.dtype == torch.float32 and query_image.max() > 1:
        query_image = query_image / 255.0

    # Binarize masks
    support_mask = (support_mask > 0).float()
    query_mask = (query_mask > 0).float()

    # Convert to numpy for overlay
    sup_img_np = TF.to_pil_image(support_image).convert("RGB")
    qry_img_np = TF.to_pil_image(query_image).convert("RGB")
    
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    sup_img_np = to_tensor(sup_img_np)
    qry_img_np = to_tensor(qry_img_np)

    def overlay_red(image, mask, alpha=0.6):
        """Overlay red on the mask region."""
        red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
        return image * (1 - mask * alpha) + red * (mask * alpha)

    support_overlay = overlay_red(sup_img_np, support_mask)
    query_overlay = overlay_red(qry_img_np, query_mask)

    # Convert back to PIL for display
    from torchvision.transforms.functional import to_pil_image
    support_overlay_pil = to_pil_image(support_overlay)
    query_overlay_pil = to_pil_image(query_overlay)
    support_img_pil = to_pil_image(sup_img_np)
    query_img_pil = to_pil_image(qry_img_np)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(support_img_pil)
    axs[0, 0].set_title('Support Image')
    axs[0, 0].axis('off')

    axs[1, 0].imshow(support_overlay_pil)
    axs[1, 0].set_title('Support + Mask (Red)')
    axs[1, 0].axis('off')

    axs[0, 1].imshow(query_img_pil)
    axs[0, 1].set_title('Query Image')
    axs[0, 1].axis('off')

    axs[1, 1].imshow(query_overlay_pil)
    axs[1, 1].set_title('Query + Mask (Red)')
    axs[1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(support_overlay_path), exist_ok=True)
    os.makedirs(os.path.dirname(query_overlay_path), exist_ok=True)

    support_overlay_pil.save(support_overlay_path)
    query_overlay_pil.save(query_overlay_path)
    
    support_img_pil.save(support_img_path)
    query_img_pil.save(query_img_path)
    
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Visualization with red mask saved to {save_path}")

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
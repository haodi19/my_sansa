
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.backbones.sem_hieradet import AdaptFormerAdapter
from sam2.utils.misc import load_video_frames_from_data

def weighted_dice_loss(prediction, target_seg, weighted_val: float = 1.0, reduction: str = "sum", eps: float = 1e-8):
    target_seg = (target_seg == 1).float()  # B, H, W
    n, h, w = target_seg.shape
    prediction = prediction.reshape(-1, h, w)  # B, H, W
    target_seg = target_seg.reshape(-1, h, w)
    prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(-1, h * w)  # B, H*W
    target_seg = target_seg.reshape(-1, h * w)
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
    loss = loss * weighted_val
    if reduction == "sum":
        loss = loss.sum() / n
    elif reduction == "mean":
        loss = loss.mean()
    return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weighted_val: float = 1.0, reduction: str = "sum"):
        super(WeightedDiceLoss, self).__init__()
        self.weighted_val = weighted_val
        self.reduction = reduction

    def forward(self, prediction, target_seg):
        return weighted_dice_loss(prediction, target_seg, self.weighted_val, self.reduction)
    
def weighted_bce_loss(
    prediction,
    target_seg,
    weighted_val: float = 1.0,
    reduction: str = "sum",
    eps: float = 1e-8
):  
    if prediction.dim() == 4:
        prediction = prediction.squeeze(1)
    if target_seg.dim() == 4:
        target_seg = target_seg.squeeze(1)

    target_seg = (target_seg == 1).float()
    n, h, w = target_seg.shape
    
    prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(-1, h * w)
    target_seg = target_seg.reshape(-1, h * w)

    bce = F.binary_cross_entropy_with_logits(prediction, target_seg, reduction="none")
    loss = bce * weighted_val

    if reduction == "sum":
        loss = loss.mean(1).sum() / n
    elif reduction == "mean":
        loss = loss.mean()
    return loss

class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, reduction: str = "sum"):
        super(CombinedBCEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.reduction = reduction

    def forward(self, prediction, target_seg, return_components=False):
        dice = weighted_dice_loss(prediction, target_seg, weighted_val=1.0, reduction=self.reduction)
        bce = weighted_bce_loss(prediction, target_seg, weighted_val=1.0, reduction=self.reduction)
        total = self.dice_weight * dice + self.bce_weight * bce
        if return_components:
            return total, dice.detach(), bce.detach()
        else:
            return total

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
                    {'params': model.sam2.image_encoder.trunk.blocks[46].adapter.parameters()},
                    {'params': model.sam2.image_encoder.trunk.blocks[47].adapter.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.parameters()},
                    # {'params': model.sam2.image_encoder.trunk.blocks[3].mlp.layers[0].parameters()},
                    # {'params': model.sam2.image_encoder.trunk.abcd.parameters()},
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
                if 'adapter' not in name:
                # if 'trunk' not in name:
                # if 'trunk.blocks.3.mlp.layers.0' not in name:
                # if 'abcd' not in name:
                    param.requires_grad = False

        # # 单独解冻 Hiera trunk 中的 Adapter 参数
        # for name, module in model.sam2.image_encoder.trunk.named_modules():
        #     if isinstance(module, AdaptFormerAdapter):
        #         for param in module.parameters():
        #             param.requires_grad = True

            

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
    def forward(self, x, s_x, s_y, y_m, cat_idx=None, priors=None):
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
            # import pdb
            # pdb.set_trace()
            # qry_feats/sup_feats: list,多尺度特征
            # qry_feats[0]: torch.Size([65536, 1, 32])
            # qry_feats[1]: torch.Size([16384, 1, 64])
            # qry_feats[2]: torch.Size([4096, 1, 256])
            # qry_feats[3]: torch.Size([16384, 1, 64])
            # qry_sizes: [(256, 256), (128, 128), (64, 64)]
  
            # add support prompt - gt mask
            sup_fg = s_mask[:, 0, ...].unsqueeze(1)  # b, 1, h, w
            
            (sup_fg_preds, sup_fg_obj_ptrs, sup_fg_mem_feats, sup_fg_mem_poss) = self.sam2.add_new_mask_batch(
                sup_feats, sup_sizes, sup_fg
            )  # support fg gt memory, b, 64, 32, 

            # import pdb
            # pdb.set_trace()
            # self.visualize_mask_on_image(s_x, sup_fg_preds, save_path='vis_result.png')

            # propagate prompted frames (直接用SAM2的propagate_in_video_batch)
            sup_mask = F.interpolate(s_mask[:, 0, ...].unsqueeze(1).float(), size=qry_sizes[-1], mode='nearest')
            low_res_masks, output_query, pix_feat_with_mem = self.sam2.propagate_in_video_batch_mine(
                qry_feats, qry_poss, qry_sizes,
                sup_fg_mem_feats, sup_fg_mem_poss, sup_fg_preds, sup_fg_obj_ptrs,
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
            

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.backbones.sem_hieradet import AdaptFormerAdapter
from sam2.utils.misc import load_video_frames_from_data
from transformers import CLIPTokenizer, CLIPTextModel


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
                if 'adapter' not in name:
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
            
            sup_fg[valid_mask] = 0  # 1, shot, H, W
            sup_fg_preds, sup_fg_obj_ptrs, sup_fg_mem_feats, sup_fg_mem_poss = self.sam2.add_new_mask_batch(
                ref_feat, ref_sizes, sup_fg
            )
            
            text_features = None
            if self.use_text_prompt:
                text_features = self.encode_class_names(class_name)
                text_features = self.text_fc(text_features)  # 输出为 [bs, 256]

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
                # slice features for current frame
                qry_feat = [f[:,j*b:(j+1)*b] for f in tgt_feat]
                qry_pos = [p[:,j*b:(j+1)*b] for p in tgt_pos]
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
                sup_fg[valid_mask] = 0 
                # update memory using predicted mask
                # pred_mask_up = F.interpolate(output_query.unsqueeze(1), size=ref_sizes[-1], mode='nearest')
                # pred_mask_up = pred_mask_up.detach()
                fg_preds, obj_ptrs, mem_feats_new, mem_pos_new = self.sam2.add_new_mask_batch(
                    qry_feat, tgt_sizes, sup_fg
                )

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
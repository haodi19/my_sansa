import argparse
import torch
import torch.nn as nn
import os
import os.path as osp

from model import FSSAM  # 或你实际的模型import
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from util.fss_dataset import FSSDataset

from util import config  # 第二份代码的config工具

import torch.nn.functional as F
from model import FSSAM, FSSAM5s, simple_fssam

import time

# ====== 第二份代码的cfg合并逻辑 ======
def get_cfg():
    parser = argparse.ArgumentParser(description='FSSAM in HSNet Test Framework')
    parser.add_argument('--arch', type=str, default='FSSAM')
    parser.add_argument('--logpath', type=str, default='test_logs')
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--config', type=str, default='config/coco/coco_split3_resnet50.yaml',
                        help='config file')
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--num_refine', type=int, default=3,
                        help='number of memory refinement')
    parser.add_argument('--ver_refine', type=str, default="v1",
                        help='version of memory refinement')
    parser.add_argument('--ver_dino', type=str, default="dinov2_vitb14", choices=["dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="version of dino")
    parser.add_argument('--episode', help='number of test episodes', type=int, default=1000)

    # 你可以根据需要添加第二份代码模型需要的其它参数
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

# ====== 第二份代码的模型获取和权重加载逻辑 ======
def get_model(args):
    model = eval(args.arch).OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr, type=args.training_type)
    model = model.cuda()
    print(args.use_original_imgsize)
    # Resume
    if args.weight:
        weight_path = args.weight
        if os.path.isfile(weight_path):
            print("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(weight_path))

    # use bfloat16 for the entire program
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    return model, optimizer

def restore_pred_mask(pred_mask, orig_size, target_size):
    """
    将预测的 pred_mask 从 padded square 恢复到原始尺寸。
    
    pred_mask: Tensor [B, 1, target_size, target_size]
    orig_size: tuple (orig_h, orig_w) — 原始query图像的高宽
    target_size: int — padded的目标尺寸
    """
    B, C, H, W = pred_mask.shape
    assert H == target_size and W == target_size

    orig_h, orig_w = orig_size
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))

    # 裁剪padding区域，居中
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    left = pad_w // 2
    pred_mask_cropped = pred_mask[:, :, top:top+new_h, left:left+new_w]

    # 插值还原到原始大小
    pred_mask_restored = F.interpolate(pred_mask_cropped, size=(orig_h, orig_w), mode='bilinear', align_corners=True)

    return pred_mask_restored  # shape: [B, 1, orig_h, orig_w]

# ====== 测试流程，数据和评估部分用第一份代码 ======
def test(model, dataloader, nshot, args):
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    model.eval()
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        # 这里用第二份代码的推理方式
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                support_imgs = batch['support_imgs']  # [B, shot, C, H, W]
                support_masks = batch['support_masks']  # [B, shot, 1, H, W]
                query_img = batch['query_img']  # [B, C, H, W]
                # 你可能还需要 batch['query_mask'], batch['class_id'] 等

                output, _ = model(
                    s_x=support_imgs,
                    s_y=support_masks,
                    x=query_img,
                    y_m=None,  # 评估时不需要gt
                    cat_idx=batch['class_id'] if 'class_id' in batch else None,
                    priors=None
                )
                # import pdb
                # pdb.set_trace()
                # visualize_fewshot_seg(support_imgs[0], support_masks, query_img, output.cpu(), save_path='output/fewshot_vis2.png')
                # output: [B, H, W] 或 [B, 1, H, W]
                if output.dim() == 4 and output.size(1) == 1:
                    output = output[:, 0]  # squeeze channel
                pred_mask = torch.sigmoid(output)
                pred_mask = (pred_mask > 0.5).float()
                # 保证 pred_mask 和 batch['query_mask'] 形状一致
                # pred_mask = F.interpolate(pred_mask.unsqueeze(1), size=batch['query_mask'].shape[-2:], mode='bilinear', align_corners=True)
                pred_mask = restore_pred_mask(pred_mask.unsqueeze(1), orig_size=batch['query_mask'].shape[-2:], target_size=1024)
                pred_mask = pred_mask.squeeze(1)

        assert pred_mask.size() == batch['query_mask'].size()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(
                batch['support_imgs'], batch['support_masks'],
                batch['query_img'], batch['query_mask'],
                pred_mask, batch['class_id'], idx,
                area_inter[1].float() / area_union[1].float()
            )

    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou

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
    # Squeeze and move to CPU
    support_image = support_image[0].cpu()
    support_mask = support_mask[0][0].cpu().float()
    query_image = query_image[0].cpu()
    query_mask = query_mask[0].cpu().float()

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


if __name__ == '__main__':
    args = get_cfg()
    Logger.initialize(args, training=False)
    # logger = Logger.logger

    # Model initialization (第二份代码的方式)
    model, optimizer = get_model(args)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization (第一份代码的方式)
    FSSDataset.initialize(img_size=1024, datapath=args.data_root, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.data_set, args.batch_size_val, args.nworker, args.split, 'test', args.shot)

    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.shot, args)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.split, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
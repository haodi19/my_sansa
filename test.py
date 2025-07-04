import os
import sys
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
from visdom import Visdom
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image

from tensorboardX import SummaryWriter

from model import FSSAM, FSSAM5s, simple_fssam

from util import dataset
from util import transform_new as transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, get_save_path, \
    is_same_model, fix_bn, sum_list, check_makedirs
import matplotlib.pyplot as plt

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
val_manual_seed = 123
setup_seed(val_manual_seed, True)
seed_array = [321]
val_num = len(seed_array)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='FSSAM')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/coco/coco_split3_resnet50.yaml',
                        help='config file')  # coco/coco_split0_resnet50.yaml
    parser.add_argument('--num_refine', type=int, default=3,
                        help='number of memory refinement')
    parser.add_argument('--ver_refine', type=str, default="v1",
                        help='version of memory refinement')
    parser.add_argument('--ver_dino', type=str, default="dinov2_vitb14", choices=["dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="version of dino")
    parser.add_argument('--episode', help='number of test episodes', type=int, default=1000)
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):
    model = eval(args.arch).OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr, type = args.training_type)

    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.weight:
        # weight_path = osp.join(args.snapshot_path, args.weight)
        weight_path = args.weight
        if os.path.isfile(weight_path):
            logger.info("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(weight_path))

    # ========================================
    # use bfloat16 for the entire program
    # ========================================
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer


class ImageMaskTransform:
    def __init__(self, img_size, mean, std):
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            # ToTensor255(),  # 会自动转为 [0,1] 且变成 C×H×W
            transforms.Normalize(mean=mean, std=std)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.PILToTensor(),  # 不除以255，输出 LongTensor（如果是L模式）
        ])

    def __call__(self, image_np, mask_np):
        # image_np: np.ndarray, (H, W, 3), float32 or uint8
        # mask_np: np.ndarray, (H, W), uint8 (values: 0 or 255)

        # 先转为 PIL.Image
        image_pil = Image.fromarray(image_np.astype(np.uint8))  # RGB
        mask_pil = Image.fromarray(mask_np.astype(np.uint8))    # L mode

        image = self.img_transform(image_pil)  # Tensor, float, normalized
        mask = self.mask_transform(mask_pil).squeeze(0).long()  # Tensor, long, shape (H, W)

        return image, mask

class ToTensor255:
    def __call__(self, img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()  # shape: (C, H, W)

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    logger.info(model)

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor()
                ])
            # val_transform = ImageMaskTransform(img_size=args.val_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor()
                ])
        if args.data_set == 'pascal' or args.data_set == 'coco':
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                       data_list=args.val_list, transform=val_transform, mode='val',
                                       ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

    # ========================================
    # Test one batch first to warmup
    # Global autocast needs to cache the conversion of fp32->bfp16
    # ========================================
    validate(val_loader, model, 321, args.episode, warmup=True)

    # ----------------------  VAL  ----------------------
    start_time = time.time()
    FBIoU_array = np.zeros(val_num)
    mIoU_array = np.zeros(val_num)
    pIoU_array = np.zeros(val_num)
    for val_id in range(val_num):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id + 1, val_num, val_seed))
        fb_iou, miou, piou = validate(val_loader, model, val_seed, args.episode)
        FBIoU_array[val_id], mIoU_array[val_id], pIoU_array[val_id] = \
            fb_iou, miou, piou

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nTotal running time: {}'.format(total_time))
    print('Seed0: {}'.format(val_manual_seed))
    print('Seed:  {}'.format(seed_array))
    print('mIoU:  {}'.format(np.round(mIoU_array, 4)))
    print('FBIoU: {}'.format(np.round(FBIoU_array, 4)))
    print('pIoU:  {}'.format(np.round(pIoU_array, 4)))
    print('-' * 43)
    print('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {}'.format(seed_array[mIoU_array.argmax()],
                                                                         seed_array[FBIoU_array.argmax()],
                                                                         seed_array[pIoU_array.argmax()]))
    print('Best_mIoU: {:.4f} \t Best_FBIoU: {:.4f} \t Best_pIoU: {:.4f}'.format(
        mIoU_array.max(), FBIoU_array.max(), pIoU_array.max()))
    print('Mean_mIoU: {:.4f} \t Mean_FBIoU: {:.4f} \t Mean_pIoU: {:.4f}'.format(
        mIoU_array.mean(), FBIoU_array.mean(), pIoU_array.mean()))


def validate(val_loader, model, val_seed, episode, warmup=False):
    if not warmup:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()  # final
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if args.data_set == 'pascal':
        test_num = 1000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = episode if not warmup else 1000
        split_gap = 20

    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    setup_seed(val_seed, args.seed_deterministic)

    pos_weight = torch.ones([1]).cuda() * 2
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.eval()
    end = time.time()
    val_start = end
    
    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))
    iter_num = 0

    for e in range(db_epoch):
        for i, (input, target, s_input, s_mask, subcls, ori_label, class_name) in enumerate(val_loader):
            if iter_num == 1 and warmup: break

            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)

            priors = None

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    start_time = time.time()
                    output, priors = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls, priors=priors, class_name=class_name)
                    # output: torch.Size([1, 1024, 1024])
                    model_time.update(time.time() - start_time)
                    # visualize_fewshot_seg(s_input[0], s_mask, input, output.cpu(), save_path='output/fewshot_vis.png')
                    # import pdb
                    # pdb.set_trace()
                    if args.ori_resize:
                        output = F.interpolate(output.unsqueeze(0), size=ori_label.size()[-2:], mode='bilinear', align_corners=True)
                        output = output.squeeze(0)
                        target = ori_label.long()

                    # output = output.unsqueeze(0)
                    output = F.interpolate(output.unsqueeze(0), size=target.size()[1:], mode='bilinear', align_corners=True)
                    output = output.squeeze(0)
                    label = target.clone()
                    label[label == 255] = 0
                    loss = criterion(output, label.float())
        
            output = torch.sigmoid(output)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            subcls = subcls[0].cpu().numpy()[0]
            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            class_intersection_meter[subcls] += intersection[1]
            class_union_meter[subcls] += union[1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            remain_iter = test_num / args.batch_size_val - iter_num
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            if ((i + 1) % round((test_num / 100)) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              remain_time=remain_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))
    if not warmup:
        val_time = time.time() - val_start

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)

        class_iou_class = []
        class_miou = 0
        for i in range(len(class_intersection_meter)):
            class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
            class_iou_class.append(class_iou)
            class_miou += class_iou

        class_miou = class_miou * 1.0 / len(class_intersection_meter)
        logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))  # final
        logger.info('<<<<<<< Novel Results <<<<<<<')
        for i in range(split_gap):
            logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))

        logger.info('FBIoU---Val result: FBIoU {:.4f}.'.format(mIoU))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou_f {:.4f}.'.format(i, iou_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

        return mIoU, class_miou, iou_class[1]






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
    main()

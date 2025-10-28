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
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler

from tensorboardX import SummaryWriter

from model import FSSAM, FSSAM5s, simple_fssam

from common import utils
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from util.fss_dataset import FSSDataset
from util import config

import torch.distributed as dist
import torch.nn.functional as F

from util.util import check_makedirs, fix_bn, fss_collate_fn, get_model_para_number, get_save_path, poly_learning_rate, setup_seed
# from util.util import AverageMeter as OldAverageMeter

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.autograd.set_detect_anomaly(True)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='FSSAM')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_vgg.yaml')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_model(args):
    model = eval(args.arch).OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr, type=args.training_type)

    if hasattr(model, 'freeze_modules'):
        model.freeze_modules(model, type=args.training_type)

    if args.distributed:
        # Initialize Process Group
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    else:
        model = model.cuda()
    
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        # resume_path = '/hdd0/ljn/new_sam2/my_fssam/exp/coco/simple_fssam/split0/large_sem_384_24_adapter_newtrans2/snapshot/train_epoch_10_0.4855.pth'
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                Logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                Logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():
                Logger.info("=> no checkpoint found at '{}'".format(resume_path))


    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        Logger.info('Number of Parameters: %d' % (total_number))
        Logger.info('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
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

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    left = pad_w // 2
    pred_mask_cropped = pred_mask[:, :, top:top+new_h, left:left+new_w]

    pred_mask_restored = F.interpolate(pred_mask_cropped, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
    return pred_mask_restored  # shape: [B, 1, orig_h, orig_w]

# === 检查函数 ===
def check_param_consistency(model, optimizer):
    # 收集 optimizer 中的参数id
    optimizer_params = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer_params.add(id(p))
    
    inconsistent = []
    for name, param in model.named_parameters():
        in_optim = id(param) in optimizer_params
        if param.requires_grad and not in_optim:
            inconsistent.append((name, "requires_grad=True but not in optimizer"))
        elif not param.requires_grad and in_optim:
            inconsistent.append((name, "requires_grad=False but in optimizer"))
    
    if inconsistent:
        print("⚠️ Inconsistencies found:")
        for name, msg in inconsistent:
            print(f" - {name}: {msg}")
    else:
        print("✅ All parameters consistent!")

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def test(model, dataloader, nshot, args):
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    model.eval()
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                support_imgs = batch['support_imgs']  # [B, shot, C, H, W]
                support_masks = batch['support_masks']  # [B, shot, 1, H, W]
                query_img = batch['query_img']  # [B, C, H, W]
                output, _ = model(
                    s_x=support_imgs,
                    s_y=support_masks,
                    x=query_img,
                    y_m=None,
                    cat_idx=batch['class_id'] if 'class_id' in batch else None,
                    priors=None
                )
                if output.dim() == 4 and output.size(1) == 1:
                    output = output[:, 0]
                pred_mask = torch.sigmoid(output)
                pred_mask = (pred_mask > 0.5).float()
                if not args.ori_resize:
                    pred_mask = restore_pred_mask(pred_mask.unsqueeze(1), orig_size=batch['query_mask'].shape[-2:], target_size=1024)
                pred_mask = pred_mask.squeeze(1)
        assert pred_mask.size() == batch['query_mask'].size()
        
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        if main_process():
            average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)
            
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(
                batch['support_imgs'], batch['support_masks'],
                batch['query_img'], batch['query_mask'],
                pred_mask, batch['class_id'], idx,
                area_inter[1].float() / area_union[1].float()
            )
    if main_process():
        average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    return miou, fb_iou

def train(train_loader, val_loader, model, optimizer, epoch, scaler, args, best_miou):
    # batch_time = AverageMeter(train_loader.dataset)
    # data_time = AverageMeter(train_loader.dataset)
    # main_loss_meter = AverageMeter(train_loader.dataset)
    # dice_loss_meter = AverageMeter(train_loader.dataset)
    # bce_loss_meter = AverageMeter(train_loader.dataset)
    # loss_meter = AverageMeter(train_loader.dataset)
    average_meter = AverageMeter(train_loader.dataset)

    if args.fix_bn:
        model.apply(fix_bn)

    end = time.time()
    max_iter = args.epochs * len(train_loader)
    
    if main_process():
        print('Warmup: {}'.format(args.warmup))

    for i, batch in enumerate(train_loader):
        model.train()
        batch = utils.to_cuda(batch)
        input = batch['query_img']
        target = batch['query_mask']
        s_input = batch['support_imgs']
        s_mask = batch['support_masks']
        cat_idx = batch['class_id'] if 'class_id' in batch else None
        
        query_clip_x = batch['query_clip_x']
        support_clip_features = batch['support_clip_features']
        query_img_f = batch['query_img_f']

        # data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1

        # 学习率调度（可选）
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                           index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        with autocast():
            output, main_loss, aux_loss1, aux_loss2, dice_loss_val, bce_loss_val = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, query_clip_x = query_clip_x,
                support_clip_features = support_clip_features, query_img_f = query_img_f, orig_size= torch.stack(batch['org_query_imsize'], dim=1).cpu().tolist(), cat_idx=cat_idx)
            loss = main_loss

        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        output = restore_pred_mask(output.unsqueeze(1), orig_size=batch['query_mask'].shape[-2:], target_size=1024)
        output = output.squeeze(1)

        area_inter, area_union = Evaluator.classify_prediction(output.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        if main_process():
            average_meter.write_process(i, len(train_loader), epoch, write_batch_idx=50)

        n = input.size(0)
        # main_loss_meter.update(main_loss.item(), n)
        # dice_loss_meter.update(dice_loss_val.item(), n)
        # bce_loss_meter.update(bce_loss_val.item(), n)
        # loss_meter.update(loss.item(), n)

        # batch_time.update(time.time() - end)
        end = time.time()

        # if (i + 1) % args.print_freq == 0 and main_process():
        #     Logger.info('Epoch: [{}/{}][{}/{}] '
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #                 'MainLoss {main_loss_meter.val:.4f} '
        #                 'Loss {loss_meter.val:.4f} '.format(
        #                     epoch + 1, args.epochs, i + 1, len(train_loader),
        #                     data_time=data_time,
        #                     batch_time=batch_time,
        #                     main_loss_meter=main_loss_meter,
        #                     loss_meter=loss_meter
        #                 ))

        if (i + 1) % (len(train_loader) // 10) == 0:
            # if (i + 1) == (len(train_loader) // 2) and main_process():
            #     Logger.info('==== Half-epoch Test ====')
            # elif main_process():
            #     Logger.info('==== Full-epoch Test ====')
            if main_process():
                epoch_idx = (i + 1) // (len(train_loader) // 10)
                Logger.info(f'==== {epoch_idx}-epoch Test ====')
                # Logger.info('==== Full-epoch Test ====')
            miou, fb_iou = test(model, val_loader, args.shot, args)
            if main_process():
                Logger.info('Epoch mIoU: {:.4f}, FB-IoU: {:.4f}'.format(miou, fb_iou))
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch + (i + 1) / len(train_loader)  # 记录半epoch位置
                filename = os.path.join(args.snapshot_path, f'best_model_{best_epoch}.pth')
                if main_process():
                    Logger.info('Saving best checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

    if main_process():
        average_meter.write_result('Training', epoch)
        
    avg_loss = utils.mean(average_meter.loss_buf)
    trn_miou, trn_fb_iou = average_meter.compute_iou()
    
    
    return avg_loss, trn_miou, trn_fb_iou, best_miou

def main():
    global args
    args = get_parser()
    args.distributed = torch.cuda.device_count() > 1

    if main_process():
        print(args)
        Logger.initialize(args, training=True)
        
    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    if main_process():
        Logger.info("=> creating model ...")

    model, optimizer = get_model(args)

    if args.viz and main_process():
        writer = SummaryWriter(args.result_path)

    # ======== 数据加载 ==========
    FSSDataset.initialize(img_size=1024, datapath=args.data_root, use_original_imgsize=args.ori_resize)
    train_loader = FSSDataset.build_dataloader(args.data_set, args.batch_size, args.nworker, args.split, 'trn', args.shot, collate_fn = fss_collate_fn)
    val_loader = FSSDataset.build_dataloader(args.data_set, args.batch_size_val, args.nworker, args.split, 'val', args.shot, collate_fn = fss_collate_fn)

    # ======== 评估器/可视化器 ==========
    Evaluator.initialize()
    Visualizer.initialize(args.viz)

    scaler = GradScaler()
    best_miou = 0.
    best_epoch = 0
    
    # for name, param in model.named_parameters():
    #     print(f"Name: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")
    # exit(0)
    
    # 获取 optimizer 中所有参数的 id
    # optim_param_ids = set(id(p) for group in optimizer.param_groups for p in group['params'])
    # print("\n=== Trainable parameters in optimizer ===")
    # for name, param in model.named_parameters():
    #     if id(param) in optim_param_ids:
    #         print(name)
    # print("=== End of trainable parameters ===\n")
    # check_param_consistency(model, optimizer)
    # exit(0)

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, trn_miou, trn_fb_iou, best_miou = train(train_loader, val_loader, model, optimizer, epoch, scaler, args, best_miou)

        # if args.viz and main_process():
        #     writer.add_scalar('train_loss', train_loss, epoch + 1)
        
        if main_process():
            Logger.info('Epoch train_loss: {:.4f}, mIoU: {:.4f}, FB-IoU: {:.4f}'.format(train_loss,trn_miou, trn_fb_iou))

        # 每个epoch结束后测试一次
        # Logger.info('==== Epoch-end Test ====')
        # miou, fb_iou = test(model, val_loader, args.shot, args)
        # Logger.info('Epoch {} mIoU: {:.4f}, FB-IoU: {:.4f}'.format(epoch + 1, miou, fb_iou))

        # if miou > best_miou:
        #     best_miou = miou
        #     best_epoch = epoch
        #     filename = os.path.join(args.snapshot_path, 'best_model.pth')
        #     if main_process():
        #         Logger.info('Saving best checkpoint to: ' + filename)
        #         torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

if __name__ == '__main__':
    main()
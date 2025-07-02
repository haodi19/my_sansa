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

from tensorboardX import SummaryWriter

from model import FSSAM, FSSAM5s, simple_fssam

from common import utils
from data.dataset import FSSDataset
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, get_save_path, is_same_model, fix_bn, sum_list, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.autograd.set_detect_anomaly(True)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='FSSAM')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_vgg.yaml')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from util import config
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
        dist.init_process_group(backend='nccl')
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
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    total_number, learnable_number = get_model_para_number(model)
    return model, optimizer

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = torch.cuda.device_count() > 1

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    model, optimizer = get_model(args)

    if args.viz and main_process():
        writer = SummaryWriter(args.result_path)

    # ======== Replace SemData with FSSDataset ==========
    FSSDataset.initialize(img_size=400, datapath=args.data_root, use_original_imgsize=False)
    train_loader = FSSDataset.build_dataloader(args.data_set, args.batch_size, args.workers, args.split, 'trn')
    val_loader = FSSDataset.build_dataloader(args.data_set, args.batch_size_val, args.workers, args.split, 'val')

    global best_miou, best_FBiou, best_piou, best_epoch, keep_epoch, val_num
    best_miou = 0.
    best_FBiou = 0.
    best_piou = 0.
    best_epoch = 0
    keep_epoch = 0
    val_num = 0

    scaler = GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, val_loader, model, optimizer, epoch, scaler)

        if main_process() and args.viz:
            writer.add_scalar('FBIoU_train', mIoU_train, epoch_log)

        if args.evaluate and epoch % 1 == 0:
            loss_val, FBIoU, mIoU, pIoU = validate(val_loader, model)
            val_num += 1
            if main_process() and args.viz:
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('FBIoU_val', FBIoU, epoch_log)
                writer.add_scalar('mIoU_val', mIoU, epoch_log)

            if mIoU > best_miou:
                best_miou, best_FBiou, best_piou, best_epoch = mIoU, FBIoU, pIoU, epoch
                keep_epoch = 0
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_{:.4f}'.format(best_miou) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(epoch) + '_{:.4f}'.format(best_miou) + '.pth'
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

def train(train_loader, val_loader, model, optimizer, epoch, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    dice_loss_meter = AverageMeter()
    bce_loss_meter = AverageMeter()
    aux_loss_meter1 = AverageMeter()
    aux_loss_meter2 = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    if args.fix_bn:
        model.apply(fix_bn)

    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)

    for i, batch in enumerate(train_loader):
        batch = utils.to_cuda(batch)
        input = batch['query_img']
        target = batch['query_mask']
        s_input = batch['support_imgs'].squeeze(1)
        s_mask = batch['support_masks'].squeeze(1)
        subcls = batch['class_id']

        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1

        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                           index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        with autocast():
            output, main_loss, aux_loss1, aux_loss2, dice_loss_val, bce_loss_val = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls)
            loss = main_loss + args.aux_weight1 * aux_loss1 + args.aux_weight2 * aux_loss2

        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n = input.size(0)
        output = torch.sigmoid(output)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter1.update(aux_loss1.item(), n)
        aux_loss_meter2.update(aux_loss2.item(), n)
        dice_loss_meter.update(dice_loss_val.item(), n)
        bce_loss_meter.update(bce_loss_val.item(), n)
        loss_meter.update(loss.item(), n)

        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'DiceLoss {dice_loss_meter.val:.4f} '
                        'BCELoss {bce_loss_meter.val:.4f} '
                        'AuxLoss1 {aux_loss_meter1.val:.4f} '
                        'AuxLoss2 {aux_loss_meter2.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          dice_loss_meter=dice_loss_meter,
                                                          bce_loss_meter=bce_loss_meter,
                                                          aux_loss_meter1=aux_loss_meter1,
                                                          aux_loss_meter2=aux_loss_meter2,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    return main_loss_meter.avg, mIoU, mAcc, allAcc



def validate(val_loader, model, warmup=False):
    if main_process() and not warmup:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if args.data_set == 'pascal':
        test_num = 1000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = 1000
        split_gap = 20

    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    pos_weight = torch.ones([1]).cuda() * 2
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))
    iter_num = 0

    dataset = args.data_set
    sam2_type = args.sam2_type
    split = args.split
    shot = args.shot

    for e in range(db_epoch):
        for i, batch in enumerate(val_loader):
            if iter_num == 1 and warmup:
                break

            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            batch = utils.to_cuda(batch)
            input = batch['query_img']
            target = batch['query_mask']
            s_input = batch['support_imgs'].squeeze(1)
            s_mask = batch['support_masks'].squeeze(1)
            subcls = batch['class_id']
            ori_label = batch['query_mask'] if 'query_mask' in batch else target

            priors = None

            start_time = time.time()
            with torch.no_grad():
                with autocast():
                    output, priors = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls, priors=priors)
                    model_time.update(time.time() - start_time)

                    if args.ori_resize:
                        output = F.interpolate(output.unsqueeze(0), size=ori_label.size()[-2:], mode='bilinear', align_corners=True)
                        output = output.squeeze(0)
                        target = ori_label.long()

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

            if ((i + 1) % round((test_num / 100)) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
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

        if main_process():
            logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
            logger.info('<<<<<<< Novel Results <<<<<<<')
            for i in range(split_gap):
                logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))
            logger.info('FBIoU---Val result: FBIoU {:.4f}.'.format(mIoU))
            for i in range(args.classes):
                logger.info('Class_{} Result: iou {:.4f}.'.format(i, iou_class[i]))
            logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

            print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

        return loss_meter.avg, mIoU, class_miou, iou_class[1] if len(iou_class) > 1 else 0.



if __name__ == '__main__':
    main()

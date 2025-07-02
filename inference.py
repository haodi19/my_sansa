import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from model import FSSAM, FSSAM5s, simple_fssam
from util import transform_new as transform, config
from util.util import setup_seed, get_model_para_number
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description='FSS Inference')
    parser.add_argument('--config', type=str, default='config/coco/coco_split3_resnet50.yaml')
    parser.add_argument('--arch', type=str, default='FSSAM')
    parser.add_argument('--ver_dino', type=str, default="dinov2_vitb14")
    parser.add_argument('--num_refine', type=int, default=3)
    parser.add_argument('--ver_refine', type=str, default='v1')
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--support_images', nargs='+', required=True)
    parser.add_argument('--support_masks', nargs='+', required=True)
    parser.add_argument('--query_image', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    return cfg, args

def load_image_mask(image_path, mask_path, transform):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path)
    image, mask = transform(image, mask)
    return image, mask

def load_query(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image, _ = transform(image, None)
    return image

def inference():
    cfg, args = get_parser()
    setup_seed(123, True)

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transform = transform.Compose([
        transform.Resize(size=cfg.val_size),
        transform.ToTensor()
    ])

    assert len(args.support_images) == len(args.support_masks), "Support images and masks must match"

    support_imgs, support_msks = [], []
    for img_path, msk_path in zip(args.support_images, args.support_masks):
        img, msk = load_image_mask(img_path, msk_path, val_transform)
        support_imgs.append(img)
        support_msks.append(msk)
    support_imgs = torch.stack(support_imgs).cuda()
    support_msks = torch.stack(support_msks).cuda()

    query_img = load_query(args.query_image, val_transform)
    query_img = query_img.unsqueeze(0).cuda()

    model = eval(args.arch).OneModel(cfg).cuda()

    if args.weight:
        checkpoint_path = os.path.join(cfg.snapshot_path, args.weight)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            print(f"=> no checkpoint found at '{checkpoint_path}'")
    else:
        print("=> no weight specified, using randomly initialized model")

    model.eval()
    torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output, _ = model(s_x=support_imgs, s_y=support_msks, x=query_img)
        output = F.interpolate(output.unsqueeze(0), size=query_img.shape[2:], mode='bilinear', align_corners=True)
        output = torch.sigmoid(output.squeeze(0)).cpu()
        pred = (output >= 0.5).float()

    # Save or show
    pred_mask = pred[0].numpy() * 255  # single-channel
    pred_mask = pred_mask.astype(np.uint8)
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        cv2.imwrite(args.save_path, pred_mask)
        print(f"Saved prediction to {args.save_path}")
    else:
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    inference()

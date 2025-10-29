r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from transformers import CLIPImageProcessor
import cv2
from pycocotools.coco import COCO

COCO_80_CATEGORIES = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
  22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
  35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
  46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
  56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
  67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = 'val2014' if split in ['val', 'test'] else 'train2014'
        # self.base_path = os.path.join(datapath, 'COCO2014')
        self.base_path = os.path.join(datapath)
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        
        # CLIP 预处理器初始化
        self.processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
        
        # 1️⃣ 加载 COCO API
        ann_file = os.path.join(self.base_path, 'annotations', f'instances_{self.split_coco}.json')
        self.coco_api = COCO(ann_file)
        
        # 构建图像名→id映射，加速后续查找
        self.name2id = {
            img_info['file_name']: img_info['id']
            for img_info in self.coco_api.dataset['images']
        }


    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000
    
    def crop_and_clip(self, img_rgb, mask_tensor, processor=None, clip_size=224, device="cuda"):
        """
        img_rgb: numpy array [H, W, 3], RGB
        mask_tensor: torch.Tensor [H, W], 二值
        """
        ys, xs = torch.where(mask_tensor > 0)
        if ys.numel() == 0 or xs.numel() == 0:
            return None  # 空mask

        ymin, ymax = ys.min().item(), ys.max().item()
        xmin, xmax = xs.min().item(), xs.max().item()

        crop = img_rgb[ymin:ymax+1, xmin:xmax+1].copy()
        mask_crop = mask_tensor[ymin:ymax+1, xmin:xmax+1].numpy()
        crop[mask_crop == 0] = 0  # 背景置空

        # ✅ pad 到正方形
        h, w, _ = crop.shape
        size = max(h, w)
        pad_top = (size - h) // 2
        pad_bottom = size - h - pad_top
        pad_left = (size - w) // 2
        pad_right = size - w - pad_left
        crop_padded = cv2.copyMakeBorder(
            crop,
            pad_top, pad_bottom,
            pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # 2) 如果正方形仍然太小（≤3），再 pad 到恰好 4×4，避免 (1,1,3)/(2,2,3)/(3,3,3) 触发通道歧义
        h2, w2, _ = crop_padded.shape  # 这里已确保 h2==w2
        if h2 <= 3:  # 同时 w2==h2
            need = 4 - h2  # 需要增加的总边数
            # 对称分配：上/左分一半，剩余给下/右
            extra_top = need // 2
            extra_bottom = need - extra_top
            extra_left = need // 2
            extra_right = need - extra_left
            crop_padded = cv2.copyMakeBorder(
                crop_padded,
                extra_top, extra_bottom,
                extra_left, extra_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
            print("chingching", h2,w2,crop_padded.shape)
            # 现在保证为 (4,4,3)

        # ✅ 使用 CLIP processor（不改属性）
        if processor is None:
            img_f = (crop_padded.astype(np.float32) / 255.0).clip(0, 1)
            x = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
        else:
            # ⚠️ 不修改 processor 配置
            x = processor(images=crop_padded, return_tensors="pt")["pixel_values"]

        return x  # Tensor [1, 3, 224, 224]
    
    def load_rgb(self, path, size, processor=None, device="cuda"):
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

        # 可视化用的 [0,1] 浮点图
        img_f = (img_rgb.astype(np.float32) / 255.0).clip(0, 1)

        # 标准 CLIP 预处理（强烈建议）
        if processor is None:
            x = torch.from_numpy(img_f).permute(2,0,1).unsqueeze(0)  # 退化备用
        else:
            # processor 会做 mean/std 归一化
            x = processor(images=img_rgb, return_tensors="pt")["pixel_values"]  # [1,3,H,W]

        # x = x.to(device)
        return img_rgb, img_f, x
    
    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, \
            class_sample, org_qry_imsize, query_inst_masks, support_inst_masks_list = self.load_frame()
            
        # support_inst_masks_list: 两层list, 第一层shot, 第二层实例个数, 每个实例torch.Size([h, w]), 原图尺寸

        tmp_s = support_imgs
        
        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        
        if not self.use_original_imgsize:
            query_mask = self.resize_and_pad_mask(query_mask, target_size=1024, pad_value=255)
            # query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        # 🆕 对support做实例mask裁剪+clip预处理
        support_clip_features_list = []
        for s_idx, (pil_img, inst_masks) in enumerate(zip(support_imgs, support_inst_masks_list)):
            img_rgb = np.array(pil_img)  # PIL → numpy [H,W,3]
            clip_features_per_support = []
            for mask_tensor in inst_masks:
                x_clip = self.crop_and_clip(img_rgb, mask_tensor, processor=self.processor, clip_size=224, device="cuda")
                if x_clip is not None:
                    clip_features_per_support.append(x_clip)
            support_clip_features_list.append(clip_features_per_support)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            # support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks[midx] = self.resize_and_pad_mask(support_masks[midx], target_size=1024, pad_value=255)
            
        support_masks = torch.stack(support_masks)

        # === 3. CLIP preprocessing ===
        query_img_rgb, query_img_f, query_clip_x = self.load_rgb(os.path.join(self.base_path, query_name), size=224, processor = self.processor)
        support_clip_x_list = [self.load_rgb(os.path.join(self.base_path, sname), size=224, processor = self.processor)[2] for sname in support_names]
        support_clip_x = torch.cat(support_clip_x_list, dim=0)  # [shot, 3, clip_size, clip_size]
        
        # self.visualize_support_masks_and_crops(save_dir="./mask_vis",support_imgs=tmp_s ,support_inst_masks_list=support_inst_masks_list ,support_clip_features_list=support_clip_features_list, support_clip_x = support_clip_x ,prefix="debug")
    
        # # 处理 query 实例 mask
        # padded_query_inst_masks = []
        # for inst_mask in query_inst_masks:
        #     inst_mask = torch.tensor(inst_mask)
        #     if not self.use_original_imgsize:
        #         inst_mask = self.resize_and_pad_mask(inst_mask, target_size=1024, pad_value=0)
        #     padded_query_inst_masks.append(inst_mask)
        # if len(padded_query_inst_masks) > 0:
        #     query_instance_masks_tensor = torch.stack(padded_query_inst_masks)
        # else:
        #     query_instance_masks_tensor = torch.zeros((0, 1024, 1024), dtype=torch.long)

        # # 处理 support 实例 mask
        # padded_support_inst_masks = []
        # num_support_instances = []
        # for inst_list in support_inst_masks_list:
        #     inst_tensors = []
        #     for inst_mask in inst_list:
        #         inst_mask = torch.tensor(inst_mask)
        #         if not self.use_original_imgsize:
        #             inst_mask = self.resize_and_pad_mask(inst_mask, target_size=1024, pad_value=0)
        #         inst_tensors.append(inst_mask)
        #     num_support_instances.append(len(inst_tensors))
        #     if len(inst_tensors) > 0:
        #         inst_tensors = torch.stack(inst_tensors)
        #     else:
        #         inst_tensors = torch.zeros((0, 1024, 1024), dtype=torch.long)
        #     padded_support_inst_masks.append(inst_tensors)
        
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample),
                 'subcls': self.class_ids.index(class_sample),
                 
                # ✅ 新增的 clip 图像
                'query_clip_x': query_clip_x,               # [1, 3, clip_size, clip_size]
                'support_clip_x': support_clip_x,           # [shot, 3, clip_size, clip_size]
                
                'query_img_f': query_img_f,
                 
                # 🆕 Query 实例 mask
                # 'query_instance_masks': query_inst_masks,
                # 'num_query_instances': len(padded_query_inst_masks),

                # 🆕 Support 实例 mask
                # 'support_instance_masks': support_inst_masks_list,  # list[shot] of [N_i, H, W], mask为原图尺寸
                # 'num_support_instances': num_support_instances,       # list[int], 每张 support 的实例数量
                
                'support_clip_features': support_clip_features_list  # 🆕 list[list[tensor[1,3,224,224]]]
                 }

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('./splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)
        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        # Query 实例 mask
        img_id = self.name2id[query_name.split('/')[1]]
        real_coco_cat_id = COCO_80_CATEGORIES[class_sample]
        ann_ids = self.coco_api.getAnnIds(imgIds=[img_id], catIds=[real_coco_cat_id])
        anns = self.coco_api.loadAnns(ann_ids)
        query_inst_masks = [torch.from_numpy(self.coco_api.annToMask(a)).long() for a in anns]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        support_inst_masks_list = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)
            
            # 🆕 support 实例 mask
            s_img_id = self.name2id[support_name.split('/')[1]]
            real_coco_cat_id = COCO_80_CATEGORIES[class_sample]
            s_ann_ids = self.coco_api.getAnnIds(imgIds=[s_img_id], catIds=[real_coco_cat_id])
            s_anns = self.coco_api.loadAnns(s_ann_ids)
            s_inst_masks = [torch.from_numpy(self.coco_api.annToMask(a)).long() for a in s_anns]
            support_inst_masks_list.append(s_inst_masks)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize, query_inst_masks, support_inst_masks_list
    
    def resize_and_pad_mask(self, mask, target_size, pad_value=255):
        """
        mask: Tensor of shape [H, W] or [1, H, W] (values are class indices or binary)
        target_size: int, 最终输出大小
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [1, H, W]

        _, h, w = mask.shape
        scale = target_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        # resize using nearest
        mask_resized = F.interpolate(mask.unsqueeze(0).float(), size=(new_h, new_w), mode='nearest').squeeze(0).long()  # [1, new_h, new_w] → [1, H, W]

        # compute padding
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # pad with ignore index (e.g. 255)
        mask_padded = F.pad(mask_resized, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

        return mask_padded.squeeze(0)  # [H, W]
    
    def visualize_support_masks_and_crops(
        self,
        save_dir,
        support_imgs,                   # list[PIL.Image]
        support_inst_masks_list,        # list[list[Tensor[H,W]]]
        support_clip_features_list,     # list[list[Tensor[1,3,224,224]]]
        support_clip_x, 
        prefix="support"
    ):
        """
        可视化 support 实例 mask 以及对应的裁剪区域
        save_dir: 输出文件夹
        support_imgs: 原始 support 图像（PIL 图像列表）
        support_inst_masks_list: 原图上每张 support 的实例 mask
        support_clip_features_list: 每个 mask 对应的裁剪区域图像（clip processor 预处理后的）
        prefix: 输出文件名前缀
        """
        def denormalize_clip(x):
            """将 CLIP processor 输出的 pixel_values 反归一化到 [0,1]"""
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
            std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
            return (x * std + mean).clamp(0, 1)
        
        os.makedirs(save_dir, exist_ok=True)
        shot = len(support_imgs)

        for s_idx in range(shot):
            # -------- 将 PIL 转换为 numpy --------
            pil_img = support_imgs[s_idx]
            img_np = np.array(pil_img)  # [H, W, 3], RGB
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)

            masks = support_inst_masks_list[s_idx]    # list[tensor[H,W]]
            crops = support_clip_features_list[s_idx] # list[tensor[1,3,224,224]]

            crop_tensor = support_clip_x  # [1, 3, 224, 224]
            crop_tensor = denormalize_clip(crop_tensor)
            crop_np = (crop_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
            crop_save_path = os.path.join(save_dir, f"{prefix}_shot{s_idx}_clipimg.jpg")
            cv2.imwrite(crop_save_path, crop_bgr)
            
            for m_idx, mask in enumerate(masks):
                # -------- 1. 保存裁剪图像（反归一化 CLIP 特征） --------
                if m_idx < len(crops):
                    crop_tensor = crops[m_idx]  # [1, 3, 224, 224]
                    crop_tensor = denormalize_clip(crop_tensor)
                    crop_np = (crop_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
                    crop_save_path = os.path.join(save_dir, f"{prefix}_shot{s_idx}_mask{m_idx}_crop.jpg")
                    cv2.imwrite(crop_save_path, crop_bgr)

                # -------- 2. 可视化 mask --------
                mask_np = mask.cpu().numpy().astype(np.uint8)
                mask_color = np.zeros_like(img_np)
                mask_color[..., 1] = 255  # 绿色叠加 mask

                alpha = 0.5
                overlay = img_np.copy()
                overlay[mask_np == 1] = cv2.addWeighted(
                    img_np[mask_np == 1], 1 - alpha,
                    mask_color[mask_np == 1], alpha,
                    0
                )

                mask_save_path = os.path.join(save_dir, f"{prefix}_shot{s_idx}_mask{m_idx}_overlay.jpg")
                cv2.imwrite(mask_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
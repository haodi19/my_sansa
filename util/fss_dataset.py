r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from util import transform_new as transform, transform_tri, config

from util.coco import DatasetCOCO
from util.fss import DatasetFSS
from util.pascal import DatasetPASCAL

import torch
import numpy as np
class ToTensor255:
    def __call__(self, img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()  # shape: (C, H, W)

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([
                                            # transforms.Resize(size=(img_size, img_size)),
                                            # ToTensor255(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std),
                                            ResizeLongSideAndPad(size=img_size, fill=0),
                                            ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
    
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

class ResizeLongSideAndPad:
    def __init__(self, size, fill=0):
        self.size = size  # 最终目标输出大小 size x size
        self.fill = fill  # padding填充值

    def __call__(self, img):
        """
        img: PIL.Image or Tensor (C, H, W)
        返回：PIL.Image or Tensor (C, size, size)
        """
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            # torch tensor: (C, H, W)
            h, w = img.shape[-2:]

        scale = self.size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        # resize (保持比例)
        img = TF.resize(img, [new_h, new_w])

        # compute padding
        pad_h = self.size - new_h
        pad_w = self.size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # pad: (left, top, right, bottom)
        img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)

        return img
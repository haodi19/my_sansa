import random
import math
import numpy as np
import numbers
import collections
collections.Iterable = collections.abc.Iterable
import cv2

import torch

manual_seed = 123
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)


class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
        for t in self.segtransform:
            # image = t(image)
            image, label = t(image, label)
        return image, label


import time


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label


class CLAHE(object):
    # Apply Contrast Limited Adaptive Histogram Equalization to the input image.
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image, label):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        clahe_mat = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = clahe_mat.apply(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            image[:, :, 0] = clahe_mat.apply(image[:, :, 0])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        image = image.astype(np.float32)
        return image, label


class ToNumpy(object):
    # Converts torch.FloatTensor of shape (C x H x W) to a numpy.ndarray (H x W x C).
    def __call__(self, image, label):
        if not isinstance(image, torch.Tensor) or not isinstance(label, torch.Tensor):
            raise (RuntimeError("segtransform.ToNumpy() only handle torch.tensor"))

        image = image.cpu().numpy().transpose((1, 2, 0))
        if not image.dtype == np.uint8:
            image = image.astype(np.uint8)
        label = label.cpu().numpy().transpose((1, 2, 0))
        if not label.dtype == np.uint8:
            label = label.astype(np.uint8)
        return image, label


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label


class UnNormalize(object):
    # UnNormalize tensor with mean and standard deviation along channel: channel = (channel * std) + mean
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.add_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.mul_(s).add_(m)
        return image, label


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = cv2.resize(image, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return image, label
    
class ResizeWithAspectAndPad(object):
    def __init__(self, size):
        self.size = size  # 目标边长（长边缩放后，短边pad）

    def __call__(self, image, label):
        h, w = image.shape[:2]
        scale = self.size / max(h, w)

        # 缩放图像和标签
        new_h, new_w = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label_resized = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 计算padding尺寸
        pad_h = self.size - new_h
        pad_w = self.size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 使用0进行padding（可根据需求改）
        image_padded = cv2.copyMakeBorder(image_resized, pad_top, pad_bottom, pad_left, pad_right, 
                                          borderType=cv2.BORDER_CONSTANT, value=0)
        label_padded = cv2.copyMakeBorder(label_resized, pad_top, pad_bottom, pad_left, pad_right, 
                                          borderType=cv2.BORDER_CONSTANT, value=255)  # 255常用于ignore_label

        return image_padded, label_padded

class ResizeWithAspectAndPad2(object):
    def __init__(self, size):
        self.size = size  # 最终输出大小（正方形）

    def __call__(self, image, label):
        import torch.nn.functional as F
        # image: [C, H, W], float tensor
        # label: [H, W], long tensor
        assert isinstance(image, torch.Tensor) and isinstance(label, torch.Tensor)

        c, h, w = image.shape
        scale = self.size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        image = image.unsqueeze(0)  # [1, C, H, W]
        image_resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=True)
        image_resized = image_resized.squeeze(0)

        label = label.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        label_resized = F.interpolate(label, size=(new_h, new_w), mode='nearest').squeeze(0).squeeze(0).long()

        # Compute padding
        pad_h = self.size - new_h
        pad_w = self.size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad
        image_padded = F.pad(image_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        label_padded = F.pad(label_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=255)

        return image_padded, label_padded

class ToTensorAndNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        # image: HWC, uint8
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0  # [3, H, W], float32
        label = torch.from_numpy(label).long()  # [H, W]

        # normalize
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        image = (image - mean) / std

        return image, label

class test_Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = cv2.resize(image, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=(self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return image, label


class Direct_Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        test_size = self.size

        image = cv2.resize(image, dsize=(test_size, test_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label.astype(np.float32), dsize=(test_size, test_size), interpolation=cv2.INTER_NEAREST)
        return image, label


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        self.size = size
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape

        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        raw_label = label
        raw_image = image

        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        raw_pos_num = np.sum(raw_label == 1)
        pos_num = np.sum(label == 1)
        crop_cnt = 0
        while (pos_num < 0.85 * raw_pos_num and crop_cnt <= 30):
            image = raw_image
            label = raw_label
            if self.crop_type == 'rand':
                h_off = random.randint(0, h - self.crop_h)
                w_off = random.randint(0, w - self.crop_w)
            else:
                h_off = int((h - self.crop_h) / 2)
                w_off = int((w - self.crop_w) / 2)
            image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            raw_pos_num = np.sum(raw_label == 1)
            pos_num = np.sum(label == 1)
            crop_cnt += 1
        if crop_cnt >= 50:
            image = cv2.resize(raw_image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(raw_label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)

        if image.shape != (self.size[0], self.size[0], 3):
            image = cv2.resize(image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)

        return image, label


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label

class RandomHorizontalFlip2(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        # image: [C, H, W], label: [H, W]
        if random.random() < self.p:
            image = torch.flip(image, dims=[2])   # flip width (W) axis
            label = torch.flip(label, dims=[1])   # flip width (W) axis
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label

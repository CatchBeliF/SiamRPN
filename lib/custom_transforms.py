from net.config import Config
import numpy as np
import cv2
import torch


# class RandomStretch(object):继承object大类
class RandomStretch:
    def __init__(self):
        self.max_stretch = Config.scale_stretch

    def __call__(self, img):
        scale_w_ratio = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_h_ratio = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w, _ = img.shape
        new_shape = (int(scale_w_ratio * w), int(scale_h_ratio * h))
        return cv2.resize(img, new_shape, cv2.INTER_LINEAR)
        # 双线性插值法


class CenterCrop:
    # 其实是没有填充的
    def __init__(self, size, img_mean=None):
        self.size = size
        self.img_mean = img_mean

    def __call__(self, img):
        img_h, img_w, img_c = img.shape
        cx, cy = (img_w - 1) / 2, (img_h - 1) / 2  # stretch之后目标还是在中心
        xmin = cx - (self.size[1] - 1) / 2
        ymin = cy - (self.size[0] - 1) / 2
        xmax = cx + (self.size[1] - 1) / 2
        ymax = cy + (self.size[0] - 1) / 2

        def round_up(x):
            # 保证两位小数的精确四舍五入 不懂！
            return round(x + 1e-6 + 1000) - 1000

        # 记录下需要的填充 这里0.就是个细节
        left_pad = int(round_up(max(0., -xmin)))
        top_pad = int(round_up(max(0., -ymin)))
        right_pad = int(round_up(max(0., xmax - img_w + 1)))
        bottom_pad = int(round_up(max(0., ymax - img_h + 1)))

        # 采用包住原图A的方式填充，先记录下想得到图的坐标之后在A中裁剪出来
        xmin = int(round_up(xmin + left_pad))
        ymin = int(round_up(ymin + top_pad))
        xmax = int(round_up(xmax + left_pad))
        ymax = int(round_up(ymax + top_pad))

        # if left_pad==0 and top_pad==0 and right_pad==0 and bottom_pad==0:
        # 学习用函数 any
        if any([left_pad, top_pad, right_pad, bottom_pad]):
            img_unit = np.zeros((img_h + top_pad + bottom_pad, img_w + left_pad + right_pad, img_c), np.uint8)
            img_unit[top_pad:top_pad + img_h, left_pad:left_pad + img_w, :] = img
            if top_pad:
                img_unit[0:top_pad, left_pad:left_pad + img_w, :] = self.img_mean
            if bottom_pad:
                img_unit[img_h + top_pad:, left_pad:left_pad + img_w, :] = self.img_mean
            if left_pad:
                img_unit[:, 0:left_pad, :] = self.img_mean
            if right_pad:
                img_unit[:, left_pad + img_w:, :] = self.img_mean
            # 填充完之后裁剪
            img_patch = img_unit[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]

        else:
            img_patch = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]

        return img_patch


class RandomCrop:
    def __init__(self):
        pass

    def __call__(self):
        pass


class ToTensor:
    def __call__(self, img):
        # img = img.transpose(2, 0, 1) 到时候调试来看
        return torch.from_numpy(img.astype(np.float32))
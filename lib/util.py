import numpy as np
import cv2


def get_instance_img(img, bbox, size_z, size_x, img_mean):
    cx, cy, w, h = bbox
    p = (w + h) / 4
    img_length = np.sqrt((w + 2 * p) * (h + 2 * p))
    # 算一个比例
    scale_ratio = size_z / img_length

    instance_length = img_length * size_x / size_z
    instance_img, scale_ratio_x = crop_and_pad(img, cx, cy, size_x, instance_length, img_mean)
    w_instance = w * scale_ratio_x
    h_instance = h * scale_ratio_x
    return instance_img, scale_ratio, w_instance, h_instance


def crop_and_pad(img, cx, cy, model_size, instance_length, img_mean=None):
    """

    :param img:
    :param cx:
    :param cy:
    :param model_size:      最终需要的尺寸
    :param instance_length: 裁剪出来的尺寸
    :param img_mean:
    :return:
    """
    img_h, img_w, img_c = img.shape
    xmin = cx - (instance_length - 1) / 2
    ymin = cy - (instance_length - 1) / 2
    xmax = cx + (instance_length - 1) / 2
    ymax = cy + (instance_length - 1) / 2

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
    # if any([left_pad, top_pad, right_pad, bottom_pad]):
    #     img_unit = np.zeros((img_h + top_pad + bottom_pad, img_w + left_pad + right_pad, img_c), np.uint8)
    #     img_unit[top_pad:top_pad + img_h, left_pad:left_pad + img_w, :] = img
    #     if left_pad:
    #         img_unit[:, 0:left_pad, :] = img_mean
    #     if top_pad:
    #         img_unit[0:top_pad, left_pad:, :] = img_mean
    #     if right_pad:
    #         img_unit[top_pad:, left_pad + img_w:, :] = img_mean
    #     if bottom_pad:
    #         img_unit[img_h + top_pad:, left_pad:left_pad + img_w, :] = img_mean
    # 上面是你自己写的 等空了看一下正确与否
    if any([left_pad, top_pad, right_pad, bottom_pad]):
        img_unit = np.zeros((img_h + top_pad + bottom_pad, img_w + left_pad + right_pad, img_c), np.uint8)
        img_unit[top_pad:top_pad + img_h, left_pad:left_pad + img_w, :] = img
        if top_pad:
            img_unit[0:top_pad, left_pad:left_pad + img_w, :] = img_mean
        if bottom_pad:
            img_unit[img_h+top_pad:, left_pad:left_pad + img_w, :] = img_mean
        if left_pad:
            img_unit[:, 0:left_pad , :] = img_mean
        if right_pad:
            img_unit[:, left_pad + img_w:, :] = img_mean
        # 填充完之后裁剪
        img_patch = img_unit[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
        # 关于这里加1前面不加1 详见笔记
        # 这里的尺寸有些差1(400,401,3)因为是小数点进位的缘故应该影响不大

    else:
        img_patch = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    # 这里加一个判断如果已经是model_size就不用resize
    if not np.array_equal(model_size, instance_length):
        img = cv2.resize(img_patch, (model_size, model_size))
    else:
        img = img_patch
    scale_ratio = model_size / img_patch.shape[0]

    return img, scale_ratio


def round_up(x):
    # 保证两位小数的精确四舍五入 不懂！
    return round(x + 1e-6 + 1000) - 1000

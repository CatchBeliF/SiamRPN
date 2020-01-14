import numpy as np
import cv2
import torch
import time
import matplotlib.pyplot as plt


# 为了到时候get_exemplar_img的时候也要填充 另写一个函数
def get_exemplar_img(img, bbox, size_z, img_mean=None):
    cx, cy, w, h = bbox
    p = (w + h) / 4
    img_length = np.sqrt((w + 2 * p) * (h + 2 * p))
    scale_ratio_z = size_z / img_length
    exemplar_img, _ = crop_and_pad(img, cx, cy, size_z, img_length, img_mean)
    return exemplar_img, scale_ratio_z, img_length


def get_instance_img(img, bbox, size_z, size_x, img_mean):
    cx, cy, w, h = bbox
    p = (w + h) / 2
    img_length = np.sqrt((w + p) * (h + p))
    # 算一个比例
    scale_ratio = size_z / img_length
    instance_length = img_length * size_x / size_z
    instance_img, scale_ratio_x = crop_and_pad(img, cx, cy, size_x, instance_length, img_mean)
    w_instance = w * scale_ratio
    h_instance = h * scale_ratio
    return instance_img, scale_ratio, w_instance, h_instance


def crop_and_pad(img, cx, cy, model_size, instance_length, img_mean=None):
    img_h, img_w, img_c = img.shape
    xmin = cx - (instance_length - 1) / 2
    ymin = cy - (instance_length - 1) / 2
    xmax = xmin + instance_length - 1
    ymax = ymin + instance_length - 1

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
            img_unit[0:top_pad, left_pad:left_pad + img_w, :] = img_mean
        if bottom_pad:
            img_unit[img_h + top_pad:, left_pad:left_pad + img_w, :] = img_mean
        if left_pad:
            img_unit[:, 0:left_pad, :] = img_mean
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


def generate_anchor(total_stride, base_size, anchor_scales, anchor_ratios, score_map_size):
    """
        anchors: cx,cy,w,h  这里的cx cy 是相对于图片中心的相对位置
    """
    anchor_num = len(anchor_scales) * len(anchor_ratios)  # 每个位置的锚框数量
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = np.square(base_size)
    count = 0

    for ratio in anchor_ratios:
        w_scaled0 = int(np.sqrt(size / ratio))
        h_scaled0 = int(w_scaled0 * ratio)
        for scale in anchor_scales:
            w_scaled = w_scaled0 * scale
            h_scaled = h_scaled0 * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = w_scaled
            anchor[count, 3] = h_scaled
            count += 1
    # 把一个位置上的anchor配置复制到所有位置上
    anchor = np.tile(anchor, score_map_size * score_map_size).reshape(-1, 4)
    shift = (score_map_size // 2) * total_stride
    xx, yy = np.meshgrid([-shift + total_stride * x for x in range(score_map_size)],
                         [-shift + total_stride * y for y in range(score_map_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def box_transform(anchor, box):
    anchor_x = anchor[:, :1]
    anchor_y = anchor[:, 1:2]
    anchor_w = anchor[:, 2:3]
    anchor_h = anchor[:, 3:]
    box_x, box_y, box_w, box_h = box
    # 论文里的公式
    dx = (box_x - anchor_x) / anchor_w
    dy = (box_y - anchor_y) / anchor_h
    dw = np.log(box_w / anchor_w)
    dh = np.log(box_h / anchor_h)
    # 返回原始锚框和box的真正的距离
    regression_target = np.hstack((dx, dy, dw, dh))
    return regression_target


def compute_iou(anchor, box):
    # 下面为了使维度相对应
    if np.array(anchor).ndim == 1:
        anchor = np.array(anchor)[None, :]  # 相当于0维加一个维度
    else:
        anchor = np.array(anchor)
    if np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    box = np.tile(box, (anchor.shape[0], 1))
    # iou计算方式是两个框覆盖面积除以两个框总共不重合的面积和
    anchor_x0 = anchor[:, :1] - (anchor[:, 2:3] - 1) / 2
    anchor_y0 = anchor[:, 1:2] - (anchor[:, 3:] - 1) / 2
    anchor_x1 = anchor[:, :1] + (anchor[:, 2:3] - 1) / 2
    anchor_y1 = anchor[:, 1:2] + (anchor[:, 3:] - 1) / 2

    box_x0 = box[:, :1] - (box[:, 2:3] - 1) / 2
    box_y0 = box[:, 1:2] - (box[:, 3:] - 1) / 2
    box_x1 = box[:, :1] + (box[:, 2:3] - 1) / 2
    box_y1 = box[:, 1:2] + (box[:, 3:] - 1) / 2

    # 下面经过修改 max要写np.max 要有axis控制维度
    # 坐标轴y轴正方向向下
    x0 = np.max([anchor_x0, box_x0], axis=0)
    y0 = np.max([anchor_y0, box_y0], axis=0)
    x1 = np.min([anchor_x1, box_x1], axis=0)
    y1 = np.min([anchor_y1, box_y1], axis=0)
    # 直接取绝对值显然不可
    # overlap_w = abs(x0 - x1)
    # overlap_h = abs(y0 - y1)
    overlap_w = np.max([(x1 - x0 + 1), np.zeros(x0.shape)], axis=0)
    overlap_h = np.max([(y1 - y0 + 1), np.zeros(y0.shape)], axis=0)
    overlap_area = overlap_h * overlap_w
    anchor_area = (anchor_x1 - anchor_x0 + 1) * (anchor_y1 - anchor_y0 + 1)
    box_area = (box_x1 - box_x0 + 1) * (box_y1 - box_y0 + 1)
    iou = overlap_area / (anchor_area + box_area - overlap_area + 1e-6)
    # 加1e-6防止分母为0
    return iou


def add_box_img(img, boxes, color=(0, 255, 0), temp=False):
    """
    :param img:
    :param boxes:  cx,cy,w,h  这里的cx，cy是相对中心点的相对位置
    :param color:
    """
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    img = img.copy()
    img_cx = (img.shape[1] - 1) / 2
    img_cy = (img.shape[0] - 1) / 2
    for box in boxes:
        # 换成以左上角为原点的坐标
        left_top_corner = [img_cx + box[0] - box[2] / 2 + 0.5, img_cy + box[1] - box[3] / 2 + 0.5]
        right_bottom_corner = [img_cx + box[0] + box[2] / 2 - 0.5, img_cy + box[1] + box[3] / 2 - 0.5]
        left_top_corner[0] = np.clip(left_top_corner[0], 0, img.shape[1])
        right_bottom_corner[0] = np.clip(right_bottom_corner[0], 0, img.shape[1])
        left_top_corner[1] = np.clip(left_top_corner[1], 0, img.shape[0])
        right_bottom_corner[1] = np.clip(right_bottom_corner[1], 0, img.shape[0])
        img = cv2.rectangle(img, (int(left_top_corner[0]), int(left_top_corner[1])),
                            (int(right_bottom_corner[0]), int(right_bottom_corner[1])),
                            color, 2)
    if temp:
        xmin = int((img_cx + boxes[:, 0] - boxes[:, 2] / 2 + 0.5).min())
        ymin = int((img_cy + boxes[:, 1] - boxes[:, 3] / 2 + 0.5).min())
        xmax = int((img_cx + boxes[:, 0] + boxes[:, 2] / 2 - 0.5).max())
        ymax = int((img_cy + boxes[:, 1] + boxes[:, 3] / 2 - 0.5).max())
        if (xmax - xmin + 1) > (ymax - ymin + 1):
            delta = xmax - xmin - ymax + ymin
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax + delta), color=(0, 0, 255), thickness=2)
            assert (xmax - xmin + 1) == (ymax + delta - ymin + 1)
        else:
            delta = ymax - ymin - xmax + xmin
            img = cv2.rectangle(img, (xmin, ymin), (xmax + delta, ymax), color=(0, 0, 255), thickness=2)
            assert (ymax - ymin + 1) == (xmax + delta - xmin + 1)
    return img


def get_topK_box(cls_score, pred_regression, anchors, topk=2):
    reg_offset = pred_regression.cpu().detach().numpy()
    scores, index = torch.topk(cls_score, topk, dim=0)
    index = index.view(-1).cpu().detach().numpy()  # debug时候看下数据转换 array([ 523,  162, 1606])
    topk_offset = reg_offset[index, :]
    anchors = anchors[index, :]
    pred_box = box_transform_use_reg_offset(anchors, topk_offset)
    return pred_box


def box_transform_use_reg_offset(anchors, offsets):
    """
    用预测的偏移量计算出box的cx,cy,w,h
    """
    anchor_cx = anchors[:, :1]
    anchor_cy = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    offsets_x, offsets_y, offsets_w, offsets_h = offsets[:, :1], offsets[:, 1:2], \
                                                 offsets[:, 2:3], offsets[:, 3:]
    box_cx = anchor_w * offsets_x + anchor_cx
    box_cy = anchor_h * offsets_y + anchor_cy
    box_w = anchor_w * np.exp(offsets_w)
    box_h = anchor_h * np.exp(offsets_h)
    boxes = np.stack([box_cx, box_cy, box_w, box_h], axis=2)
    # 对stack函数做了笔记 若box_cx.shape=(3,1) axis=2之后为（3,1,4）
    return boxes


def ajust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


# 画特征图

def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))

# def use_others_model(model):
#     model_ = model['model']
#     model_ = {k.replace('featureExtract.0', 'feature_conv1'): v for k, v in model_.items()}
#     model_ = {k.replace('featureExtract.1', 'feature_bn1'): v for k, v in model_.items()}
#     model_ = {k.replace('featureExtract.4', 'feature_conv2'): v for k, v in model_.items()}
#     model_ = {k.replace('featureExtract.5', 'feature_bn2'): v for k, v in model_.items()}
#     model_ = {k.replace('featureExtract.8', 'feature_conv3'): v for k, v in model_.items()}
#     model_ = {k.replace('featureExtract.9', 'feature_bn3'): v for k, v in model_.items()}
#     model_ = {k.replace('feature_bn11', 'feature_conv4'): v for k, v in model_.items()}
#     model_ = {k.replace('feature_bn12', 'feature_bn4'): v for k, v in model_.items()}
#     model_ = {k.replace('feature_bn14', 'feature_conv5'): v for k, v in model_.items()}
#     model_ = {k.replace('feature_bn15', 'feature_bn5'): v for k, v in model_.items()}
#     model_ = {k.replace('conv_r1', 'conv_reg1'): v for k, v in model_.items()}
#     model_ = {k.replace('conv_r2', 'conv_reg2'): v for k, v in model_.items()}
#     model['model'] = model_
#     return model


def use_others_model(model):
    model_ = model['model']
    model_ = {k.replace('sharedFeatExtra.0', 'feature_conv1'): v for k, v in model_.items()}
    model_ = {k.replace('sharedFeatExtra.1', 'feature_bn1'): v for k, v in model_.items()}
    model_ = {k.replace('sharedFeatExtra.4', 'feature_conv2'): v for k, v in model_.items()}
    model_ = {k.replace('sharedFeatExtra.5', 'feature_bn2'): v for k, v in model_.items()}
    model_ = {k.replace('sharedFeatExtra.8', 'feature_conv3'): v for k, v in model_.items()}
    model_ = {k.replace('sharedFeatExtra.9', 'feature_bn3'): v for k, v in model_.items()}
    model_ = {k.replace('feature_bn11', 'feature_conv4'): v for k, v in model_.items()}
    model_ = {k.replace('feature_bn12', 'feature_bn4'): v for k, v in model_.items()}
    model_ = {k.replace('feature_bn14', 'feature_conv5'): v for k, v in model_.items()}
    model_ = {k.replace('feature_bn15', 'feature_bn5'): v for k, v in model_.items()}
    model['model'] = model_
    return model

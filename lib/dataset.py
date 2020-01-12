from torch.utils import data
import os
import numpy as np
from glob import glob
from net.config import Config
import cv2
from torchvision.transforms import transforms
from lib.custom_transforms import RandomStretch
from lib.util import crop_and_pad, generate_anchor, box_transform, compute_iou


class Getdata(data.Dataset):
    """
    返回 batch_size张 模板图片 搜索图片 分类标签 regression_target
    """
    def __init__(self, sequence_name, video_dir, z_transforms, x_transforms, meta_data=None, training=True):
        self.sequence_name = sequence_name
        self.video_dir = video_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        self.meta_data = meta_data
        self.meta_data = {x[0]: x[1] for x in meta_data}
        self.num = len(sequence_name) if not training else 2 * len(sequence_name)
        # 对trajs做处理小于2个的直接删掉
        for img_name in self.meta_data.keys():
            trajs = self.meta_data[img_name]
            for i in list(trajs.keys()):
                if len(trajs[i]) < 2:
                    del trajs[i]
        self.training = training  # 不知道啥作用
        self.max_shift = Config.max_shift
        self.anchors = generate_anchor(Config.total_stride, Config.anchor_base_size, Config.anchor_scales,
                                       Config.anchor_ratio, Config.train_map_size)

    def __getitem__(self, idx):
        all_idx = np.arange(self.num)
        np.random.shuffle(all_idx)
        all_idx = np.insert(all_idx, 0, idx, 0)  # 把idx放在all_idx的第一位
        for idx in all_idx:
            # 先选一个序列idx,然后序列里选一个目标trkid,然后选一帧traj
            idx = idx % len(self.sequence_name)
            img_name = self.sequence_name[idx]
            trajs = self.meta_data[img_name]
            # 刚刚上面对trajs做了处理，如果有个序列的值都被删除了。则需考虑
            # assert len(trajs.keys()) > 0
            if len(trajs.keys()) == 0:
                continue
            trkid = np.random.choice(list(trajs.keys()))
            traj = trajs[trkid]
            assert len(traj) > 1, "sequence_name: {}".format(img_name)
            exemplar_idx = np.random.choice(list(range(len(traj))))
            exemplar_path = glob(os.path.join(self.video_dir, img_name,
                                              traj[exemplar_idx] + ".{:02d}.gt*.jpg".format(trkid)))[0]
            exemplar_img = cv2.imread(exemplar_path)
            # 一开始选的 图片大小都是(500, 500, 3)
            # cv2默认为BGR顺序，一般软件用RGB
            # exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
            exemplar_gt_w, exemplar_gt_h, exemplar_w, exemplar_h = self.common_fuc(exemplar_path)
            # 这里的gt_w,h是相对于500的图片
            # 接下来的这些并不懂，但为了训练效果好吧,过滤掉一些特殊案例
            img_ratio = min(exemplar_gt_w / exemplar_gt_h, exemplar_gt_h / exemplar_gt_w)
            img_scale = exemplar_gt_w * exemplar_gt_h / (exemplar_h * exemplar_w)
            if not Config.scale_range[0] <= img_scale < Config.scale_range[1]:
                continue
            if not Config.ratio_range[0] <= img_ratio < Config.ratio_range[1]:
                continue
            # -------------------------------------------------
            low_idx = max(0, exemplar_idx - Config.frame_range)
            high_idx = min(exemplar_idx + Config.frame_range, len(traj))
            # 下面加入采样权重是为了每个都能平等的选到
            weights = self.sample_weights(exemplar_idx, low_idx, high_idx, Config.sample_type)
            instance_name = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:high_idx], p=weights)
            instance_path = glob(os.path.join(self.video_dir, img_name, instance_name + ".{:02d}.gt*.jpg".format(trkid)))[0]
            instance_gt_w, instance_gt_h, instance_w, instance_h = self.common_fuc(instance_path)
            # 接下来的这些并不懂，但为了训练效果好吧,过滤掉一些特殊案例
            img_ratio = min(instance_gt_h / instance_gt_w, instance_gt_w / instance_gt_h)
            img_scale = instance_gt_h * instance_gt_w / (instance_h * instance_w)
            if not Config.scale_range[0] <= img_scale < Config.scale_range[1]:
                continue
            if not Config.ratio_range[0] <= img_ratio < Config.ratio_range[1]:
                continue
            # -------------------------------------------------
            instance_img = cv2.imread(instance_path)
            # instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
            # 下面这些做数据增强  记得调试的时候来这里看一下 数据增强以后的图片样子
            if np.random.rand(1) < Config.gray_ratio:
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)  # RGB转灰度
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)  # 灰度转RGB
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)

            # 接下来对图像再做处理，一方面尺寸另一方面中心偏移，相当于数据增强了
            exemplar_img = self.z_transforms(exemplar_img)
            # 这里模板图片的处理暂且这样
            instance_img, instance_gt_w_new, instance_gt_h_new, instance_gt_cx, instance_gt_cy = self.all_fuc(
                instance_img,
                instance_gt_w,
                instance_gt_h)
            instance_img = self.x_transforms(instance_img)

            # 下面计算box和anchor之间的iou
            # 细节anchor是[7.,7.,104.,32.] 而box有小数点 所以要round四舍五入
            box = np.array(list(map(round, [instance_gt_cx, instance_gt_cy, instance_gt_w_new, instance_gt_h_new])))
            regression_target, cls_label_map = self.compute_target(self.anchors, box)
            # 这里的anchor,box都是列表
            return exemplar_img, instance_img, regression_target, cls_label_map.astype(np.int64), box
            # 这里cls_label_map是根据iou的 dtype = float64

    def common_fuc(self, img_path):
        img_gt_w, img_gt_h, img_w, img_h = float(img_path.split('/')[-1].split('.')[2].split('_')[-1]), \
                                           float(img_path.split('/')[-1].split('.')[4].split('_')[-1]), \
                                           float(img_path.split('/')[-1].split('.')[6].split('_')[-1]), \
                                           float(img_path.split('/')[-1].split('.')[8].split('_')[-1])

        return img_gt_w, img_gt_h, img_w, img_h

    def sample_weights(self, center, low_idx, high_idx, sample_type='uniform'):
        if Config.non_local:
            weights = list(range(low_idx, high_idx))
            weights.remove(center)
            weights = weights[0: -Config.inst_num]
        else:
            weights = list(range(low_idx, high_idx))
            weights.remove(center)
        weights = np.array(weights)
        if sample_type == 'linear':
            weights = abs(weights - center)
        if sample_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        if sample_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def all_fuc(self, img, gt_w, gt_h):
        # randomstretch
        h, w, _ = img.shape
        transform = transforms.Compose([
            RandomStretch()
        ])
        img = transform(img)
        scale_h, scale_w, _ = img.shape
        scale_h_ratio, scale_w_ratio = scale_h / h, scale_w / w
        gt_h, gt_w = gt_h * scale_h_ratio, gt_w * scale_w_ratio
        # 中心偏移 一开始疑问偏移之后裁剪又是在中心 但其实目标还是在中心，只是cx,cy变了
        cx, cy = (scale_w - 1) / 2, (scale_h - 1) / 2
        cx_add_shift = cx + np.random.randint(-self.max_shift, self.max_shift)
        cy_add_shift = cy + np.random.randint(-self.max_shift, self.max_shift)
        instance_img, scale = crop_and_pad(img, cx_add_shift, cy_add_shift, Config.train_instance_size, Config.train_instance_size)
        # gt_h_new, gt_w_new = scale * gt_h, scale * gt_w  这句话不用加 一直混乱 是从randomstrech中裁剪 框的宽高不变的
        cx_new, cy_new = cx - cx_add_shift, cy - cy_add_shift
        # 这里的cx_new cy_new是相对中心的偏移量
        return instance_img, gt_w, gt_h, cx_new, cy_new

    def compute_target(self, anchors, box):
        regression_target = box_transform(anchors, box)
        iou = compute_iou(anchors, box)
        # 这里要不要flatten()先保留 不用加否则损失函数那里会报错
        pos_index = np.where(iou > Config.iou_pos_threshold)[0]
        neg_index = np.where(iou < Config.iou_neg_threshold)[0]
        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label

    def __len__(self):
        return self.num

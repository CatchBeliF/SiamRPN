from got10k.trackers import Tracker
from net.net_siamrpn import SiameseAlexnet
import torch
import numpy as np
from lib.util import get_exemplar_img, get_instance_img, box_transform_use_reg_offset, generate_anchor,use_others_model
from net.config import Config
from torchvision.transforms import transforms
from lib.custom_transforms import ToTensor
import torch.nn.functional as F
from lib.util import use_others_model


class SiamRPNTracker(Tracker):
    def __init__(self, model_path):
        super(SiamRPNTracker, self).__init__(
            name='siamrpn', is_deterministic=True
        )
        self.model = SiameseAlexnet()
        checkpoint = torch.load(model_path)  # checkpoint.keys()=dict_keys(['epoch', 'model', 'optimizer'])
        if Config.use_others:
            checkpoint = use_others_model(checkpoint)
        print("----------------loading trained model--------------------\n")
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print("----------------finish loading---------------------------\n")
        self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])

    def init(self, frame, bbox):
        """
        :param frame: current frame
        :param bbox:  left top corner, w,h
        :return:
        """
        bbox = np.array([bbox[0] + (bbox[2] - 1) / 2,
                         bbox[1] + (bbox[3] - 1) / 2,
                         bbox[2],
                         bbox[3]])

        frame = np.array(frame)
        self.center_pos = bbox[:2]
        self.target_sz = bbox[2:]
        replace = False
        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(frame.shape[:2]) < 0.004:
            replace = True
        self.instance_size = Config.track_instance_size if not replace else 287
        self.track_map_size = int((self.instance_size - Config.exemplar_size) / Config.total_stride) + 1
        # 提前算好搜索图片的大小
        self.target_sz_h, self.target_sz_w = bbox[3], bbox[2]
        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        self.anchors = generate_anchor(Config.total_stride, Config.anchor_base_size, Config.anchor_scales,
                                       Config.anchor_ratio, self.track_map_size)
        hanning = np.hanning(self.track_map_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), Config.anchor_num)

        img_mean = np.mean(frame, axis=(0, 1))
        exemplar_img, scale_ratio, _ = get_exemplar_img(frame, bbox, Config.exemplar_size, img_mean)
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        self.model.track_init(exemplar_img.permute(0, 3, 1, 2).cuda())

    def update(self, frame):
        """
        :param frame:
        :return: bbox:[xmin, ymin, w,h] 最后定位目标的框
        """
        frame = np.array(frame)
        bbox = np.hstack((self.center_pos, self.target_sz))
        img_mean = np.mean(frame, axis=(0, 1))
        instance_img, scale_detection, _, _ = get_instance_img(frame, bbox, Config.exemplar_size, self.instance_size,
                                                               img_mean)
        instance_img = self.transforms(instance_img)[None, :, :, :]
        pred_cls, pred_reg = self.model.track_next(instance_img.permute(0, 3, 1, 2).cuda())
        pred_cls = pred_cls.reshape(-1, 2, Config.anchor_num * self.track_map_size * self.track_map_size).permute(0,
                                                                                                                  2,
                                                                                                                  1)
        # torch.Size([32, 1805, 2])
        pred_reg = pred_reg.reshape(-1, 4, Config.anchor_num * self.track_map_size * self.track_map_size).permute(0,
                                                                                                                  2,
                                                                                                                  1)
        delta = pred_reg.cpu().detach().numpy().squeeze()  # squeeze的作用是当batch_size是1的时候处理的
        pred_box = box_transform_use_reg_offset(self.anchors, delta).squeeze()
        pred_cls_score = F.softmax(pred_cls.squeeze(), dim=1).data[:, 1].cpu().detach().numpy()

        # 这句不一样 记得来调试
        # 以下来自论文
        def overall_scale(w, h):
            p = 1 / 2 * (w + h)
            s = (w + p) * (h + p)
            return np.sqrt(s)

        def max_ratio(ratio):
            return np.maximum(ratio, 1 / ratio)

        # r represents (the proposal’s or the forward frame's)ratio of height and width
        # s represent the overall scale of (the proposal or the forward frame)
        s = overall_scale(pred_box[:, 2], pred_box[:, 3])
        s1 = overall_scale(self.target_sz_w * scale_detection, self.target_sz_h * scale_detection)
        # 之前对这里有疑惑 但是要搞清楚根号对应127
        s_final = max_ratio(s / s1)
        r = pred_box[:, 2] / pred_box[:, 3]
        r1 = self.target_sz_w / self.target_sz_h
        r_final = max_ratio(r / r1)
        penalty = np.exp(-(s_final * r_final - 1) * Config.penalty_k)
        # 加惩罚
        pred_cls_score = pred_cls_score * penalty
        # 加余弦窗
        pred_cls_score = pred_cls_score * (1 - Config.window_influence) + self.window * Config.window_influence
        highest_score_id = np.argmax(pred_cls_score)
        target = pred_box[highest_score_id, :] / scale_detection
        # 学习率更新
        lr = pred_cls_score[highest_score_id] * Config.track_lr
        # clip boundary
        res_x = np.clip(target[0] + self.center_pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.center_pos[1], 0, frame.shape[0])
        res_w = np.clip(self.target_sz_w * (1 - lr) + target[2] * lr,
                        Config.min_scale * self.origin_target_sz[0],
                        Config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz_h * (1 - lr) + target[3] * lr,
                        Config.min_scale * self.origin_target_sz[1],
                        Config.max_scale * self.origin_target_sz[1])
        # 在这里 你的self.origin_target_sz一开始是self.target_sz但是他会更新

        # update state
        self.center_pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        self.target_sz_w = res_w
        self.target_sz_h = res_h
        bbox = np.array([res_x, res_y, res_w, res_h])
        box = np.array([
            np.clip(bbox[0] - bbox[2] / 2, 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1] - bbox[3] / 2, 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64)
        ])

        return box


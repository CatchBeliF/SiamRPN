import numpy as np


class Config:
    exemplar_size = 127
    crop_size = 500
    train_instance_size = 255
    track_instance_size = 271
    pairs_imgs = 2
    scale_range = (0.001, 0.7)
    ratio_range = (0.1, 10)
    frame_range = 100
    sample_type = 'uniform'
    gray_ratio = 0.25
    scale_stretch = 0.15
    max_shift = 12
    anchor_base_size = 8
    anchor_scales = np.array([8, ])
    anchor_ratio = np.array([0.33, 0.5, 1, 2, 3])
    total_stride = 8
    train_map_size = int((train_instance_size - exemplar_size) / total_stride) + 1
    iou_pos_threshold = 0.6
    iou_neg_threshold = 0.3
    train_batch_size = 32
    valid_batch_size = 8
    train_ratio = 0.99
    train_num_workers = 4
    valid_num_workers = 4
    seed = 6666
    anchor_num = len(anchor_ratio) * len(anchor_scales)
    epoch = 50
    num_pos = 16
    num_neg = 48
    lamda = 1
    # 学习率本来你想单纯的设定一个 学习率决定了参数移动到最优值的速度快慢
    start_lr = 3e-2
    end_lr = 1e-5
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    momentum = 0.9  # 加速
    weight_decay = 0.0005  # 权重衰减 防止过拟合
    clip = 10
    ohem_pos = False
    ohem_neg = False
    pretrained_model = "/home/cbf/pycharmprojects/siamrpn/model/alexnet.pth"
    fix_former_3_layers = True
    log_dir = '/home/cbf/pycharmprojects/siamrpn/data/logs'
    show_interval = 3
    topk = 5
    save_interval = 1
    non_local = False
    inst_num = 5
    penalty_k = 0.055
    window_influence = 0.42
    track_lr = 0.295
    min_scale = 0.1
    max_scale = 10
    use_others = False

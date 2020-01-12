import os
import pickle
from torchvision.transforms import transforms
from lib.custom_transforms import RandomStretch, CenterCrop, ToTensor
from net.config import Config
from lib.dataset import Getdata
from torch.utils.data import DataLoader
import multiprocessing as mp
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from net.net_siamrpn import SiameseAlexnet
from lib.loss import rpn_smoothL1, rpn_cross_entropy_balance
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from lib.visual import visual
import torch.nn.functional as F
from lib.util import add_box_img, compute_iou, get_topK_box, box_transform_use_reg_offset, ajust_learning_rate
from collections import OrderedDict


def train(video_dir, model_path=None, vis_port=None, init=None):
    # 得到处理过的视频序列
    meta_data_path = os.path.join(video_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    sequence_name = [x[0] for x in meta_data]
    # 划分训练集和验证集
    train_sequences, valid_sequences = train_test_split(sequence_name, test_size=1 - Config.train_ratio,
                                                        random_state=Config.seed)

    train_z_transforms = transforms.Compose([
        # RandomStretch(),
        CenterCrop((Config.exemplar_size, Config.exemplar_size)),
        ToTensor(),
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        # RandomStretch(),
        CenterCrop((Config.exemplar_size, Config.exemplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # get train dadaset
    train_dataset = Getdata(train_sequences, video_dir, train_z_transforms, train_x_transforms, meta_data,
                            training=True)
    anchors_show = train_dataset.anchors
    # get valid dataset
    valid_dataset = Getdata(valid_sequences, video_dir, valid_z_transforms, valid_x_transforms, meta_data,
                            training=False)
    # 创建dataloader容器
    train_dataloader = DataLoader(train_dataset, batch_size=Config.train_batch_size * torch.cuda.device_count(),
                                  shuffle=True, num_workers=Config.train_num_workers * torch.cuda.device_count(),
                                  pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=Config.train_batch_size * torch.cuda.device_count(),
                                  shuffle=True, num_workers=Config.valid_num_workers * torch.cuda.device_count(),
                                  pin_memory=True, drop_last=True)
    # 只有一张显卡所以这torch.cuda.device_count()是1

    # 创建summary writer
    if not os.path.exists(Config.log_dir):
        os.mkdir(Config.log_dir)
    summary_writer = SummaryWriter(Config.log_dir)
    # 可视化
    if vis_port:
        vis = visual()

    model = SiameseAlexnet()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.lr, momentum=Config.momentum,
                                weight_decay=Config.weight_decay)

    start_epoch = 1
    # load model
    if model_path and init:
        print("init training with checkpoint %s" % model_path + '\n')
        print('---------------------------------------------------------- -- \n')
        # 这里load的是整个模型，包括网络、优化方法等等
        checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            # 这里加载的是网络的pred_cls_score
            model.load_state_dict(checkpoint['model'])
        # 换个方式加载
        else:
            model_dict = model.state_dict()  # state_dict返回的是整个网络的状态的字典
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
        del checkpoint
        # 只有执行完下面这句，显存才会在Nvidia-smi中释放
        torch.cuda.empty_cache()
        print("finish initing checkpoint! \n")
    elif model_path and not init:
        print("loading the previous checkpoint %s" % model_path + '\n')
        print('------------------------------------------------------------- \n')
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()
        print("finish loading previous checkpoint! \n")
    elif not model_path and Config.pretrained_model:
        print("load pre-trained checkpoint %s" % Config.pretrained_model + '\n')
        print('---------------------------- ------------------------------- \n')
        checkpoint = torch.load(Config.pretrained_model)
        model_dict = model.state_dict()
        temp = 'feature'
        temp0 = 'num_batches_tracked'
        i = 0
        for key in list(model_dict.keys()):
            if (temp in key) and (temp0 not in key):
                model_dict[key] = checkpoint[list(checkpoint.keys())[i]]
                i += 1
        # checkpoint = {k.replace('features.features', 'sharedFeature'): v for k, v in checkpoint.items()}
        # model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        print("finish loading pre-trained model \n")

    # 训练的时候前3个层的参数是固定的
    def freeze_layers(model):
        for layer in model.former_3_layers:
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  # 由于参数固定，所以这层的bn相当于是测试模式
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            elif isinstance(layer, nn.Conv2d):
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            elif isinstance(layer, nn.MaxPool2d):
                continue
            elif isinstance(layer, nn.ReLU):
                continue
            else:
                raise KeyError("something wrong in fixing 3 layers \n")
            # print("fixed layers:  \n", layer)

    for epoch in range(start_epoch, Config.epoch + 1):
        print("start epoch{} \n".format(epoch))
        train_loss = []
        model.train()
        if Config.fix_former_3_layers:
            if torch.cuda.device_count() > 1:
                freeze_layers(model.module)  # 多GPU 在DataParallel上要加module
            else:
                freeze_layers(model)
        loss_cls = 0
        loss_reg = 0
        for i, data in enumerate(tqdm(train_dataloader)):
            # torch.Size([32, 127, 127, 3]) torch.Size([32, 1805, 4])  torch.Size([32, 1805, 1]) torch.Size([32, 4])
            exemplar_img, instance_img, regression_target, cls_label_map, gt_original_box = data
            regression_target, cls_label_map = regression_target.cuda(), cls_label_map.cuda()
            pred_cls_score, pred_reg_anchor = model(exemplar_img.permute(0, 3, 1, 2).cuda(),
                                                    instance_img.permute(0, 3, 1, 2).cuda())
            # torch.Size([1, 320, 19, 19]) torch.Size([1, 640, 19, 19])
            pred_cls_score = pred_cls_score.reshape(-1, 2, Config.anchor_num * Config.train_map_size
                                                    * Config.train_map_size).permute(0, 2, 1)
            pred_reg_anchor = pred_reg_anchor.reshape(-1, 4, Config.anchor_num * Config.train_map_size
                                                      * Config.train_map_size).permute(0, 2, 1)
            # torch.Size([32, 1805, 2]) torch.Size([32, 1805, 4])
            cls_loss = rpn_cross_entropy_balance(pred_cls_score, cls_label_map, Config.num_pos, Config.num_neg,
                                                 ohem_pos=Config.ohem_pos, ohem_neg=Config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_reg_anchor, regression_target, cls_label_map, Config.num_pos, ohem=None)
            loss = cls_loss + Config.lamda * reg_loss

            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clip)  # 防止梯度爆炸
            optimizer.step()
            step = (epoch - 1) * len(train_dataloader) + i
            summary_writer.add_scalars('train',
                                       {'cls_loss': cls_loss.data.item(), 'reg_loss': reg_loss.data.item(),
                                        'total_loss': loss.data.item()}, step)
            train_loss.append(loss.detach().cpu())
            loss_cls += cls_loss.detach().cpu().numpy()
            loss_reg += reg_loss.detach().cpu().numpy()
            if (i + 1) % Config.show_interval == 0:
                tqdm.write("\n [epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f, lr: %.2e"
                           % (epoch, i, loss_cls / Config.show_interval, loss_reg / Config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_reg = 0
                loss_cls = 0
                # 可视化
                if vis_port:
                    topk = Config.topk
                    exemplar_img = exemplar_img[0].cpu().numpy()
                    instance_img = instance_img[0].cpu().numpy()
                    vis.plot_img(exemplar_img.transpose(2, 0, 1), win=1, name='exemplar_img')
                    gt_box_old = gt_original_box[0].detach().cpu().numpy()
                    gt_box = gt_box_old.reshape(1, 4).astype(np.float32)
                    # show gt box
                    img_box = add_box_img(instance_img, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=2, name='instance_img')
                    # show anchor with max score
                    cls_pred = F.softmax(pred_cls_score, dim=2)[0, :, 1]
                    scores, index = torch.topk(cls_pred, k=topk)
                    img_box = add_box_img(instance_img, anchors_show[index.cpu()])
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=3, name='pre_cls_score_max_anchor')

                    cls_pred = F.softmax(pred_cls_score, dim=2)[0, :, 1]
                    # 对这句话有了更深的理解 懂了dim=2和[0,：,1]的意思
                    topk_box = get_topK_box(cls_pred, pred_reg_anchor[0], anchors_show, topk=topk)
                    img_box = add_box_img(instance_img, topk_box.squeeze())
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=4, name='pred_reg_anchor_max_score')

                    # show anchor and detected box with max iou
                    iou = compute_iou(anchors_show, gt_box).flatten()
                    index = np.argsort(iou)[-topk:]
                    img_box = add_box_img(instance_img, anchors_show[index])
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=5, name='anchor_max_iou')

                    # detected box
                    regress_offset = pred_reg_anchor[0].cpu().detach().numpy()
                    topk_offset = regress_offset[index, :]
                    anchors_det = anchors_show[index, :]
                    pred_box = box_transform_use_reg_offset(anchors_det, topk_offset)
                    img_box = add_box_img(instance_img, pred_box.squeeze())
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=6, name='box_max_iou')

        train_loss = np.mean(train_loss)
        # finish training an epoch ,start validing
        valid_loss = []
        model.eval()  # 用训练好的模型测试
        for i, data in enumerate(tqdm(valid_dataloader)):
            exemplar_img, instance_img, regression_target, cls_label_map, _ = data
            regression_target, cls_label_map = regression_target.cuda(), cls_label_map.cuda()
            exemplar_img = exemplar_img.permute(0, 3, 1, 2).cuda()
            instance_img = instance_img.permute(0, 3, 1, 2).cuda()
            pred_cls_score, pred_reg_anchor = model(exemplar_img, instance_img)
            pred_cls_score = pred_cls_score.reshape(-1, 2, Config.anchor_num * Config.train_map_size
                                                    * Config.train_map_size).permute(0, 2, 1)
            pred_reg_anchor = pred_reg_anchor.reshape(-1, 4, Config.anchor_num * Config.train_map_size
                                                      * Config.train_map_size).permute(0, 2, 1)
            cls_loss = rpn_cross_entropy_balance(pred_cls_score, cls_label_map, Config.num_pos, Config.num_neg,
                                                 ohem_pos=Config.ohem_pos, ohem_neg=Config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_reg_anchor, regression_target, cls_label_map, Config.num_pos, ohem=None)
            loss = cls_loss + Config.lamda * reg_loss
            valid_loss.append(loss.detach().cpu())
        valid_loss = np.mean(valid_loss)
        print("[epoch %2d] valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        summary_writer.add_scalars('valid', {'cls_loss': cls_loss.data.item(),
                                             'reg_loss': reg_loss.data.item(),
                                             'total_loss': loss.data.item()}, epoch)
        ajust_learning_rate(optimizer, Config.gamma)

        # save model
        if epoch % Config.save_interval == 0:
            if not os.path.exists('../data/models/'):
                os.mkdir('../data/models/')
            save_name = '../data/models/siamrpn_epoch_{}.pth'.format(epoch)
            if torch.cuda.device_count() > 1:
                new_state_dict = OrderedDict()
                for k, v in model.state_dict().items():
                    new_state_dict[k] = v
            new_state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model as:{}'.format(save_name))


if __name__ == '__main__':
    video_dir = "/home/cbf/datasets/ILSVRC2015_VID/ILSVRC2015_VID_curation"
    model_path = None
    vis_port = 8097
    init = None
    train(video_dir, model_path, vis_port, init)

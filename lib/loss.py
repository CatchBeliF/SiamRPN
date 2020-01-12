import numpy as np
import random
import torch.nn.functional as F
import torch


def rpn_cross_entropy_balance(input, target, pos_num, neg_num, ohem_pos=None, ohem_neg=None):
    """
    :param input:   torch.Size([batch_size, 19*19*5, 2])
    :param target:  torch.size([batch_size, 19*19*5, 1])
    :param pos_num:
    :param neg_num:
    :return:
    """
    loss_all = []
    for batch_idx in range(target.shape[0]):
        pos_idx = np.where(target[batch_idx].cpu() == 1)[0]  # 数组类型
        neg_idx = np.where(target[batch_idx].cpu() == 0)[0]
        min_pos = min(len(pos_idx), pos_num)
        min_neg = int(min_pos * neg_num / pos_num)  # 没有int是浮点数
        if ohem_pos:
            pass
        else:
            # 这里遇到这个问题if input.size(0) != target.size(0):
            # IndexError: dimension specified as 0 but tensor has no dimensions
            if len(pos_idx) > 0:
                pos_idx_random = random.sample(pos_idx.tolist(), min_pos)
                if len(pos_idx_random) == 1:
                    pos_loss = F.cross_entropy(input=input[batch_idx][pos_idx_random],
                                               target=target[batch_idx][pos_idx_random[0]], reduction='none')
                else:
                    pos_loss = F.cross_entropy(input=input[batch_idx][pos_idx_random],
                                               target=target[batch_idx][pos_idx_random].squeeze(), reduction='none')
            else:
                pos_loss = torch.FloatTensor([0]).cuda()[0]
        if ohem_neg:
            pass
        else:
            if len(pos_idx) > 0:
                neg_idx_random = random.sample(neg_idx.tolist(), min_neg)
                neg_loss = F.cross_entropy(input=input[batch_idx][neg_idx_random],
                                           target=target[batch_idx][neg_idx_random].squeeze(), reduction='none')
            else:
                neg_idx_random = random.sample(neg_idx.tolist(), neg_num)
                neg_loss = F.cross_entropy(input=input[batch_idx][neg_idx_random],
                                           target=target[batch_idx][neg_idx_random].squeeze(), reduction='none')
        loss = (pos_loss.mean() + neg_loss.mean()) / 2
        loss_all.append(loss)  # loss_all是列表不能用mean() 并且是tensor
    loss_all_final = torch.stack(loss_all).mean()
    return loss_all_final


def rpn_smoothL1(input, target, label, pos_num, ohem=None):
    """
    :param input:  torch.size([batch_size, 19*19*5, 4])
    :param target: torch.size([batch_size, 19*19*5, 4])
    :param label:  torch.size([batch_size, 19*19*5, 1])
    :param pos_num: 自定义的正样本个数
    :param ohem:
    :return:
    """
    all_loss = []
    for batch_idx in range(target.shape[0]):  # batch_size=32
        pos_idx = np.where(label[batch_idx].cpu() == 1)[0]
        pos_min = min(len(pos_idx), pos_num)
        if pos_min > 0:
            pos_idx = random.sample(pos_idx.tolist(), pos_min)  # pos_idx本来是数组
            loss = F.smooth_l1_loss(input[batch_idx][pos_idx], target[batch_idx][pos_idx])
        else:
            loss = torch.FloatTensor([0]).cuda()[0]  # loss = tensor(0., device='cuda:0')
        all_loss.append(loss.mean())
    final_loss = torch.stack(all_loss).mean()
    return final_loss

from torch import nn
from net.config import Config
import torch.nn.functional as F
from lib.util import draw_features


class SiameseAlexnet(nn.Module):
    def __init__(self):
        super(SiameseAlexnet, self).__init__()
        # self.sharedFeature = nn.Sequential(
        #     # conv1
        #     nn.Conv2d(3, 96, 11, stride=2),
        #     nn.BatchNorm2d(96),
        #     nn.MaxPool2d(3, stride=2),  # 对特征图进行下采样用3*3的窗口去选
        #     nn.ReLU(inplace=True),  # RELU(x)=max(0,x)
        #     # conv2
        #     NONLocalBlock2D(in_channels=96),
        #     nn.Conv2d(96, 256, 5),
        #     nn.BatchNorm2d(256),
        #     nn.MaxPool2d(3, stride=2),
        #     nn.ReLU(inplace=True),
        #     # conv3
        #     nn.Conv2d(256, 384, 3),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        #     # conv4
        #     nn.Conv2d(384, 384, 3),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        #     # conv5
        #     nn.Conv2d(384, 256, 3),
        #     nn.BatchNorm2d(256),
        # )
        # 以下做了修改，为了符合预训练模型的参数以及方便可视化
        self.feature_conv1 = nn.Conv2d(3, 96, 11, stride=2)
        self.feature_bn1 = nn.BatchNorm2d(96)
        self.MaxPool1 = nn.MaxPool2d(3, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        # self.non_local = NONLocalBlock2D(in_channels=96)
        self.feature_conv2 = nn.Conv2d(96, 256, 5)
        self.feature_bn2 = nn.BatchNorm2d(256)
        self.MaxPool2 = nn.MaxPool2d(3, stride=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.feature_conv3 = nn.Conv2d(256, 384, 3)
        self.feature_bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)

        self.feature_conv4 = nn.Conv2d(384, 384, 3)
        self.feature_bn4 = nn.BatchNorm2d(384)
        self.relu4 = nn.ReLU(inplace=True)

        self.feature_conv5 = nn.Conv2d(384, 256, 3)
        self.feature_bn5 = nn.BatchNorm2d(256)

        self.anchor_num = Config.anchor_num
        self.conv_cls1 = nn.Conv2d(256, 256 * self.anchor_num * 2, 3)
        self.conv_cls2 = nn.Conv2d(256, 256, 3)
        self.conv_reg1 = nn.Conv2d(256, 256 * self.anchor_num * 4, 3)
        self.conv_reg2 = nn.Conv2d(256, 256, 3)

        self.regress_adjust = nn.Conv2d(4 * Config.anchor_num,
                                        4 * Config.anchor_num, 1)

        self.former_3_layers = [self.feature_conv1, self.feature_bn1,
                                self.feature_conv2, self.feature_bn2,
                                self.feature_conv3, self.feature_bn3]

    def forward(self, template, instance):
        N = template.shape[0]
        # savepath = "/home/cbf/PycharmProjects/siamrpn/datas/features"
        # template_feature = self.sharedFeature(template)
        # detection_feature = self.sharedFeature(detection)
        template_feature = self.feature_conv1(template)
        template_feature = self.feature_bn1(template_feature)
        template_feature = self.MaxPool1(template_feature)
        template_feature = self.relu1(template_feature)

        template_feature = self.feature_conv2(template_feature)
        template_feature = self.feature_bn2(template_feature)
        template_feature = self.MaxPool2(template_feature)
        template_feature = self.relu2(template_feature)

        template_feature = self.feature_conv3(template_feature)
        template_feature = self.feature_bn3(template_feature)
        template_feature = self.relu3(template_feature)

        template_feature = self.feature_conv4(template_feature)
        template_feature = self.feature_bn4(template_feature)
        template_feature = self.relu4(template_feature)

        template_feature = self.feature_conv5(template_feature)
        template_feature = self.feature_bn5(template_feature)

        instance_feature = self.feature_conv1(instance)
        instance_feature = self.feature_bn1(instance_feature)
        instance_feature = self.MaxPool1(instance_feature)
        instance_feature = self.relu1(instance_feature)
        # draw_features(8, 12, instance_feature.cpu().detach().numpy(), "{}/conv1.png".format(savepath))
        # 加入了non-local
        # instance_feature = self.non_local(instance_feature)
        # draw_features(8, 12, instance_feature.cpu().detach().numpy(), "{}/conv1_nonlocal.png".format(savepath))
        # vis_feature = instance_feature
        instance_feature = self.feature_conv2(instance_feature)
        instance_feature = self.feature_bn2(instance_feature)
        instance_feature = self.MaxPool2(instance_feature)
        instance_feature = self.relu2(instance_feature)

        instance_feature = self.feature_conv3(instance_feature)
        instance_feature = self.feature_bn3(instance_feature)
        instance_feature = self.relu3(instance_feature)

        instance_feature = self.feature_conv4(instance_feature)
        instance_feature = self.feature_bn4(instance_feature)
        instance_feature = self.relu4(instance_feature)

        instance_feature = self.feature_conv5(instance_feature)
        instance_feature = self.feature_bn5(instance_feature)

        template_cls = self.conv_cls1(template_feature)  # torch.Size([32, 2560, 4, 4])
        template_cls = template_cls.view(N, 2 * Config.anchor_num, 256, 4, 4)
        template_reg = self.conv_reg1(template_feature).view(N, 4 * Config.anchor_num, 256, 4, 4)

        instance_cls = self.conv_cls2(instance_feature)
        instance_reg = self.conv_reg2(instance_feature)
        template_cls = template_cls.reshape(-1, 256, 4, 4)
        template_reg = template_reg.reshape(-1, 256, 4, 4)
        cls_size = list(instance_cls.shape)[-1]
        reg_size = list(instance_reg.shape)[-1]
        instance_cls = instance_cls.reshape(1, -1, cls_size, cls_size)
        instance_reg = instance_reg.reshape(1, -1, reg_size, reg_size)

        pred_cls = F.conv2d(instance_cls, template_cls, groups=N)
        pred_reg = F.conv2d(instance_reg, template_reg, groups=N)
        pred_reg = self.regress_adjust(pred_reg.reshape(N, 4 * Config.anchor_num,
                                                        Config.train_map_size, Config.train_map_size))

        return pred_cls, pred_reg

    def track_init(self, template):
        """
         :return: input the first frame, compute it as Fixed convolution kernel
        """
        N = template.shape[0]
        template_feature = self.feature_conv1(template)
        template_feature = self.feature_bn1(template_feature)
        template_feature = self.MaxPool1(template_feature)
        template_feature = self.relu1(template_feature)

        template_feature = self.feature_conv2(template_feature)
        template_feature = self.feature_bn2(template_feature)
        template_feature = self.MaxPool2(template_feature)
        template_feature = self.relu2(template_feature)

        template_feature = self.feature_conv3(template_feature)
        template_feature = self.feature_bn3(template_feature)
        template_feature = self.relu3(template_feature)

        template_feature = self.feature_conv4(template_feature)
        template_feature = self.feature_bn4(template_feature)
        template_feature = self.relu4(template_feature)

        template_feature = self.feature_conv5(template_feature)
        template_feature = self.feature_bn5(template_feature)

        kernel_cls = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_reg = self.conv_reg1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.kernel_cls = kernel_cls.reshape(-1, 256, 4, 4)
        self.kernel_reg = kernel_reg.reshape(-1, 256, 4, 4)

    def track_next(self, instance):
        # savepath = "/home/cbf/PycharmProjects/siamrpn"
        N = instance.shape[0]
        instance_feature = self.feature_conv1(instance)
        instance_feature = self.feature_bn1(instance_feature)
        instance_feature = self.MaxPool1(instance_feature)
        instance_feature = self.relu1(instance_feature)
        # draw_features(8, 12, instance_feature.cpu().detach().numpy(), "{}/conv1_non1.png".format(savepath))
        # 加入了non-local
        # instance_feature = self.non_local(instance_feature)
        # draw_features(8, 12, instance_feature.cpu().detach().numpy(), "{}/conv1_non2.png".format(savepath))
        instance_feature = self.feature_conv2(instance_feature)
        instance_feature = self.feature_bn2(instance_feature)
        instance_feature = self.MaxPool2(instance_feature)
        instance_feature = self.relu2(instance_feature)

        instance_feature = self.feature_conv3(instance_feature)
        instance_feature = self.feature_bn3(instance_feature)
        instance_feature = self.relu3(instance_feature)

        instance_feature = self.feature_conv4(instance_feature)
        instance_feature = self.feature_bn4(instance_feature)
        instance_feature = self.relu4(instance_feature)

        instance_feature = self.feature_conv5(instance_feature)
        instance_feature = self.feature_bn5(instance_feature)

        inst_cls = self.conv_cls2(instance_feature)
        inst_reg = self.conv_reg2(instance_feature)
        cls_size = list(inst_cls.shape)[-1]
        reg_size = list(inst_reg.shape)[-1]
        inst_cls = inst_cls.reshape(1, -1, cls_size, cls_size)
        inst_reg = inst_reg.reshape(1, -1, reg_size, reg_size)

        pred_cls = F.conv2d(inst_cls, self.kernel_cls, groups=N)
        pred_reg = F.conv2d(inst_reg, self.kernel_reg, groups=N)
        map_size = pred_reg.size()[-1]
        pred_reg = self.regress_adjust(pred_reg.reshape(N, 4 * Config.anchor_num,
                                                        map_size, map_size))

        return pred_cls, pred_reg

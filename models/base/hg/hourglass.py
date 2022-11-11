import torch
from torch import nn
from models.base.hg.layers import Conv, Hourglass, Pool, Residual, Merge


# function: 基于AMIL的Hourglass网络
class StackedHourglass(nn.Module):
    # function: 初始化
    # params:
    #   （1）k：关节点个数
    #   （2）nstack：stacked-hourglass模块的堆叠数量
    def __init__(self, k, nStack):
        super(StackedHourglass, self).__init__()
        # 1. 参数设置
        self.k = k
        self.nStack = nStack

        # 2. Pose网络初始化
        # 2.1 通道、特征尺寸对齐（batchSize*3*256*256 ==> batchSize*256*64*64）
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, 256)
        )

        # 2.2 4-Stacked_Hourglass
        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, 256, False, 0)
            ) for i in range(self.nStack)])

        # 2.3 特征提取
        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(256, 256),
                Conv(256, 256, 1, bn=True, relu=True)
            ) for i in range(self.nStack)])

        # 2.4 Pose预测
        self.preds = nn.ModuleList([Conv(256, self.k, 1, relu=False, bn=False) for i in range(self.nStack)])

        # 2.5 feature融合
        self.merge_features = nn.ModuleList([Merge(256, 256) for i in range(self.nStack - 1)])

        # 2.6 pred融合
        self.merge_preds = nn.ModuleList([Merge(self.k, 256) for i in range(self.nStack - 1)])

    # function: 前馈
    # params:
    #   （1）imgs：图像数据
    # return:
    #   （1）preds：预测结果
    def forward(self, imgs):
        x = self.pre(imgs)  # in: [bs, 3, 256, 256]; out: [bs, 256, 64, 64]

        combined_preds = []
        for i in range(self.nStack):
            hg = self.hgs[i](x)  # out: [bs, 256, 64, 64]
            feature = self.features[i](hg)  # out: [bs, 256, 64, 64]
            preds = self.preds[i](feature)  # out: [bs, k, 64, 64]
            # Record the middle preds.
            combined_preds.append(preds)
            # Combine the feature and middle prediction with original image.
            if i < self.nStack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)  # out: [bs, 256, 64, 64]
        # Stack the middle prediction into one tensor.
        preds = torch.stack(combined_preds, 1) # out: [bs, nStack, k, 64, 64]

        return preds


def hg(k, nStack=4, nograd=False):
    model = StackedHourglass(k, nStack).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model

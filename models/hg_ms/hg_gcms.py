import math
import random
import numpy as np
import torch
from torch import nn
from models.base.hg.layers import Conv, G_Conv, Hourglass, Pool, Residual, Merge


class MultiStreamHourglass(nn.Module):
    def __init__(self, k, nStack, nStream, mixType):
        super(MultiStreamHourglass, self).__init__()
        # parameters
        self.k = k
        self.nStack = nStack
        self.nStream = nStream
        self.mixType = mixType

        # prepare
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, 256)
        )

        # 4-Stacked Hourglass
        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, 256, False, 0)
            ) for sIdx in range(self.nStack)])

        # multiple stream features
        self. features = nn.ModuleList([
            nn.Sequential(
                Residual(256, self._make_divisible(256, self.nStream)),
                G_Conv(self._make_divisible(256, self.nStream), self._make_divisible(256, self.nStream), 1, grp=self.nStream, bn=False, relu=False)
            ) for sIdx in range(self.nStack)])

        # multiple stream predictions
        self.stream_preds = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    Conv(int(self._make_divisible(256, self.nStream)/self.nStream), 256, 1, bn=True, relu=True),
                    Conv(256, self.k, 1, bn=False, relu=False)
                ) for stIdx in range(self.nStream)]
            ) for sIdx in range(self.nStack)])

        # # multiple stream features
        # self.features = nn.ModuleList([
        #     nn.Sequential(
        #         Residual(256, 256*self.nStream),
        #         G_Conv(256*self.nStream, 256*self.nStream, 1, grp=self.nStream, bn=False, relu=False)
        #     ) for sIdx in range(self.nStack)])
        #
        # # multiple stream predictions
        # self.stream_preds = nn.ModuleList([
        #     nn.ModuleList([
        #         nn.Sequential(
        #             Conv(256, 256, 1, bn=True, relu=True),
        #             Conv(256, self.k, 1, bn=False, relu=False)
        #         ) for stIdx in range(self.nStream)]
        #     ) for sIdx in range(self.nStack)])

        # feature merge
        self.merge_features = nn.ModuleList([Merge(self._make_divisible(256, self.nStream), 256) for sIdx in range(self.nStack - 1)])

        # prediction merge
        self.merge_preds = nn.ModuleList([Merge(self.k, 256) for sIdx in range(self.nStack - 1)])

    def forward(self, imgs):
        x = self.pre(imgs)
        preds_ms_combined = []
        for sIdx in range(self.nStack):
            hg = self.hgs[sIdx](x)
            features = self.features[sIdx](hg)

            features_ms_array, preds_ms_array = [], []
            for stIdx in range(self.nStream):
                cNum = int(features.size(1)/self.nStream)
                features_ms = features[:, cNum*stIdx:cNum*(stIdx+1)]
                features_ms_array.append(features_ms)
                preds_ms = self.stream_preds[sIdx][stIdx](features_ms)
                preds_ms_array.append(preds_ms)
            preds_ms_combined.append(torch.stack(preds_ms_array, 1))
            preds_ms_mix = torch.mean(torch.stack(preds_ms_array, 1), 1)

            # Combine the feature and middle prediction with original image.
            if sIdx < self.nStack - 1:
                x = x + self.merge_preds[sIdx](preds_ms_mix) + self.merge_features[sIdx](features)

        return torch.stack(preds_ms_combined, 1)

    def _make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


def hg_ms(k, nStack, nStream, mixType="mean", nograd=False):
    model = MultiStreamHourglass(k, nStack, nStream, mixType).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model

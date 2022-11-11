import math
import random
import numpy as np
import torch
from torch import nn
from models.base.hg.layers import Conv, Hourglass, Pool, Residual, Merge


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
                Residual(256, 256),
                Conv(256, 256, 1, bn=True, relu=True)
            ) for sIdx in range(self.nStack)])

        # multiple stream predictions
        self.stream_preds = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    Conv(256, self.k, 1, bn=False, relu=False)
                ) for stIdx in range(self.nStream)]
            ) for sIdx in range(self.nStack)])

        # feature merge
        self.merge_features = nn.ModuleList([Merge(256, 256) for sIdx in range(self.nStack - 1)])

        # prediction merge
        self.merge_preds = nn.ModuleList([Merge(self.k, 256) for sIdx in range(self.nStack - 1)])

    def forward(self, imgs):
        x = self.pre(imgs)
        preds_ms_combined = []
        for sIdx in range(self.nStack):
            hg = self.hgs[sIdx](x)
            features = self.features[sIdx](hg)

            preds_ms_array = []
            for stIdx in range(self.nStream):
                preds_ms = self.stream_preds[sIdx][stIdx](features)
                preds_ms_array.append(preds_ms)
            preds_ms_combined.append(torch.stack(preds_ms_array, 1))
            preds_ms_mix = torch.mean(torch.stack(preds_ms_array, 1), 1)

            # Combine the feature and middle prediction with original image.
            if sIdx < self.nStack - 1:
                x = x + self.merge_preds[sIdx](preds_ms_mix) + self.merge_features[sIdx](features)

        return torch.stack(preds_ms_combined, 1)

    def _channel_expand(self, channel_num):
        return int((1. + self.split_factor * (self.nStream-3)) * channel_num)
        # return channel_num

    def _channel_switchs(self, cNum):
        mask_cNum = int(cNum * self.split_factor)
        switchsArray = []
        for sIdx in range(self.nStack):
            switchs_ms_bg = random.sample(range(0, cNum), mask_cNum * self.nStream)
            switchs_ms, switchs_comm = [[] for stIdx in range(self.nStream)], []
            commIdx = -1  # idx of item in switchs_comm
            for cIdx in range(0, cNum):
                if cIdx in switchs_ms_bg:
                    commIdx += 1
                    switchs_ms[commIdx // mask_cNum].append(cIdx)
                else:
                    switchs_comm.append(cIdx)
            switchsArray.append(switchs_ms + [switchs_ms_bg, switchs_comm])
        return switchsArray

    def _channel_mask_init(self):
        bs, c, h, w = self.maxBS, self._channel_expand(256), 64, 64  # default
        channel_masks_array = []
        for bsIdx in range(bs):
            size = [bsIdx+1, c, h, w]
            channel_masks = []
            for sIdx in range(self.nStack):
                channel_mask = []
                for stIdx in range(self.nStream):
                    channel_mask.append(self._channel_mask(sIdx, stIdx, size))
                channel_masks.append(channel_mask)
            channel_masks_array.append(channel_masks)
        return channel_masks_array

    def _channel_mask(self, sIdx, stIdx, size, perturb=True):
        switchsDist = self.channel_switchs[sIdx]
        switchs_f, switchs_bg, switchs_comm = switchsDist[stIdx], switchsDist[-2], switchsDist[-1]

        # set channel_switchs
        mask = torch.ones(size)
        mask[:, switchs_bg] = 0
        mask[:, switchs_f] = 1
        # channel perturbation
        if perturb:
            return self._channel_perturb(sIdx, stIdx, mask)
        else:
            return mask

    def _channel_perturb(self, sIdx, stIdx, mask):
        if self.perturb_factor > 0:
            switchsDist = self.channel_switchs[sIdx]
            switchs_f, switchs_bg, switchs_comm = switchsDist[stIdx], switchsDist[-2], switchsDist[-1]
            switchs_perturb = random.sample(switchs_comm, int(len(switchs_comm) * self.perturb_factor))
            mask[:, switchs_perturb] = 0
        return mask


def hg_ms(k, nStack, nStream, mixType="mean", nograd=False):
    model = MultiStreamHourglass(k, nStack, nStream, mixType).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model

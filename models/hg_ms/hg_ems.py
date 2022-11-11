import math
import random
import numpy as np
import torch
from torch import nn
from models.base.hg.layers import Conv, Hourglass, Pool, Residual, Merge


class ExpansiveStreamHourglass(nn.Module):
    def __init__(self, maxBS, k, nStack, nStream, mixType, split_factor, perturb_factor, switchRandom, switchs=None, masks=None):
        super(ExpansiveStreamHourglass, self).__init__()
        # parameters
        self.maxBS = maxBS
        self.k = k
        self.nStack = nStack
        self.nStream = nStream
        self.mixType = mixType
        self.split_factor = split_factor
        self.perturb_factor = perturb_factor
        self.switchRandom = switchRandom

        self.channel_switchs = switchs if switchs is not None else self._channel_switchs(self._channel_expand(256))
        self.channel_masks_default = masks if masks is not None else self._channel_mask_init()

        # prepare
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, self._channel_expand(128)),
            Pool(2, 2),
            Residual(self._channel_expand(128), self._channel_expand(128)),
            Residual(self._channel_expand(128), self._channel_expand(256))
        )

        # 4-Stacked Hourglass
        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, self._channel_expand(256), False, 0)
            ) for sIdx in range(self.nStack)])

        # multiple stream features
        self. features = nn.ModuleList([
            nn.Sequential(
                Residual(self._channel_expand(256), self._channel_expand(256)),
                Conv(self._channel_expand(256), self._channel_expand(256), 1, bn=True, relu=True)
            ) for sIdx in range(self.nStack)])

        # multiple stream predictions
        self.stream_preds = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    Conv(self._channel_expand(256), self.k, 1, bn=False, relu=False)
                ) for stIdx in range(self.nStream)]
            ) for sIdx in range(self.nStack)])

        # feature merge
        self.merge_features = nn.ModuleList([Merge(self._channel_expand(256), self._channel_expand(256)) for sIdx in range(self.nStack - 1)])

        # prediction merge
        self.merge_preds = nn.ModuleList([Merge(self.k, self._channel_expand(256)) for sIdx in range(self.nStack - 1)])

    def forward(self, imgs):
        x = self.pre(imgs)
        x_size = x.size()
        preds_ms_combined, features_ms_combined, features_ms_f_combined = [], [], []
        for sIdx in range(self.nStack):
            hg = self.hgs[sIdx](x)
            features = self.features[sIdx](hg)

            preds_ms_array, features_ms_array, features_ms_f_array = [], [], []
            for stIdx in range(self.nStream):
                if self.training:
                    mask = self._channel_perturb(sIdx, stIdx, self.channel_masks_default[x_size[0]-1][sIdx][stIdx])
                else:
                    mask = self.channel_masks_default[x_size[0]-1][sIdx][stIdx]
                features_ms = features * mask.to(x.device, non_blocking=True)
                features_ms_array.append(features_ms)
                preds_ms = self.stream_preds[sIdx][stIdx](features_ms)
                preds_ms_array.append(preds_ms)
                cSwitch = self.channel_switchs[sIdx][stIdx]
                features_ms_f = features_ms[:, cSwitch]
                features_ms_f_array.append(features_ms_f)
            preds_ms_combined.append(torch.stack(preds_ms_array, 1))
            preds_ms_mix = torch.mean(torch.stack(preds_ms_array, 1), 1)
            features_ms_combined.append(torch.stack(features_ms_array, 1))
            features_ms_f_combined.append(torch.stack(features_ms_f_array, 1))

            # Combine the feature and middle prediction with original image.
            if sIdx < self.nStack - 1:
                x = x + self.merge_preds[sIdx](preds_ms_mix) + self.merge_features[sIdx](features)

        return torch.stack(preds_ms_combined, 1), torch.stack(features_ms_combined, 1), torch.stack(features_ms_f_combined, 1)

    def _channel_expand(self, channel_num):
        return channel_num

    def _channel_switchs(self, cNum):
        mask_cNum = int(cNum * self.split_factor)
        switchsArray = []
        for sIdx in range(self.nStack):
            if self.switchRandom:
                switchs_ms_bg = random.sample(range(0, cNum), mask_cNum * self.nStream)
            else:
                switchs_ms_bg = [item for item in range(0, mask_cNum * self.nStream)]
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


def hg_ems(maxBS, k, nStack, nStream, mixType="mean", splitFactor=0.1, perturbFactor=0.0, switchRandom=True, switchs=None, masks=None, nograd=False):
    model = ExpansiveStreamHourglass(maxBS, k, nStack, nStream, mixType, splitFactor, perturbFactor, switchRandom, switchs, masks).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model

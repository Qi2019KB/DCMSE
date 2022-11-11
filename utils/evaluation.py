# -*- coding: utf-8 -*-
import torch


class EvaluationUtils:
    def __init__(self):
        pass

    @classmethod
    def accuracy(cls, preds, gts, pck_ref, pck_thr):
        bs, k, _ = preds.shape
        # 计算各点的相对距离
        dists, dists_ref = cls._accuracy_calDists(cls, preds, gts, pck_ref)

        # 计算error
        errs, err_sum, err_num = torch.zeros(k + 1), 0, 0
        for kIdx in range(k):
            if errs[kIdx] >= 0:  # 忽略带有-1的值
                errs[kIdx] = dists[kIdx].sum() / len(dists[kIdx])
                err_sum += errs[kIdx]
                err_num += 1
        errs[-1] = err_sum / err_num

        # 根据thr计算accuracy
        accs, acc_sum, acc_num = torch.zeros(k + 1), 0, 0
        for kIdx in range(k):
            accs[kIdx] = cls._accuracy_counting(cls, dists_ref[kIdx], pck_thr)
            if accs[kIdx] >= 0:  # 忽略带有-1的值
                acc_sum += accs[kIdx]
                acc_num += 1
        if acc_num != 0:
            accs[-1] = acc_sum / acc_num
        return errs, accs

    def _accuracy_calDists(self, preds, gts, pckRef_idxs):
        # 计算参考距离（基于数据集的参考关键点对）
        bs, k, _ = preds.shape
        dists, dists_ref = torch.zeros(k, bs), torch.zeros(k, bs)
        for iIdx in range(bs):
            norm = torch.dist(gts[iIdx, pckRef_idxs[0], 0:2], gts[iIdx, pckRef_idxs[1], 0:2])
            for kIdx in range(k):
                if gts[iIdx, kIdx, 0] > 1 and gts[iIdx, kIdx, 1] > 1:
                    dists[kIdx, iIdx] = torch.dist(preds[iIdx, kIdx, 0:2], gts[iIdx, kIdx, 0:2])
                    dists_ref[kIdx, iIdx] = torch.dist(preds[iIdx, kIdx, 0:2], gts[iIdx, kIdx, 0:2]) / norm
                else:
                    dists[kIdx, iIdx] = -1
                    dists_ref[kIdx, iIdx] = -1
        return dists, dists_ref

    def _accuracy_counting(cls, dists, thr=0.5):
        dists_plus = dists[dists != -1]
        if len(dists_plus) > 0:
            return 1.0 * (dists_plus < thr).sum().item() / len(dists_plus)
        else:
            return -1

# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
import math
from itertools import combinations as comb
# import array

from .udaap.transforms import transform_preds
from .udaap.evaluation import final_preds


class ProcessUtils:
    def __init__(self):
        pass

    @classmethod
    def coord_distance(cls, coord1, coord2):
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

    @classmethod
    def coord_avgDistance(cls, coords):
        coords_comb = comb(coords, 2)
        dist_sum, dist_n = 0., 0
        while True:
            try:
                coords_pair = next(coords_comb)
                dist = cls.coord_distance(coords_pair[0], coords_pair[1])
                dist_sum += dist
                dist_n += 1
            except StopIteration:
                break
        return dist_sum/dist_n

    @classmethod
    def coord_distances(cls, coords):
        coords_comb = comb(coords, 2)
        dists = []
        while True:
            try:
                coords_pair = next(coords_comb)
                dist = cls.coord_distance(coords_pair[0], coords_pair[1])
                dists.append(dist)
            except StopIteration:
                break
        return dists

    @classmethod
    def coord_mean(cls, coords):
        xs, ys = [item[0] for item in coords], [item[1] for item in coords]
        return [sum(xs)/len(xs), sum(ys)/len(ys)]

    @classmethod
    def box_cocoStandard(cls, bbox):
        minX, minY, width, height = bbox
        lt = [minX, minY]
        rb = [minX + width, minY + height]
        return [lt, rb]

    @classmethod
    def image_load(cls, pathname):
        return np.array(cv2.imread(pathname), dtype=np.float32)

    @classmethod
    def image_save(cls, img, pathname, compression=0):
        # 创建路径
        folderPath = os.path.split(pathname)[0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        cv2.imwrite(pathname, img, [cv2.IMWRITE_PNG_COMPRESSION, compression])  # 压缩值，0为高清，9为最大压缩（压缩时间长），默认3.

    @classmethod
    def image_resize(cls, img, kps, inpRes):
        h, w, _ = img.shape
        scale = [inpRes / w, inpRes / h]
        img = cv2.resize(img, (inpRes, inpRes))
        kps = [[kp[0] * scale[0], kp[1] * scale[1], kp[2]] for kp in kps]
        return img, kps, scale

    @classmethod
    def image_resize_mulKps(cls, img, kpsArray, inpRes):
        h, w, _ = img.shape
        scale = [inpRes / w, inpRes / h]
        img = cv2.resize(img, (inpRes, inpRes))
        kpsArray_new = []
        for kps in kpsArray:
            kpsArray_new.append([[kp[0] * scale[0], kp[1] * scale[1], kp[2]] for kp in kps])
        return img, kpsArray_new, scale

    @classmethod
    def image_colorNorm(cls, img, means, stds, useStd=False):
        if img.size(0) == 1:  # 黑白图处理
            img = img.repeat(3, 1, 1)

        for t, m, s in zip(img, means, stds):  # 彩色图处理
            t.sub_(m)  # 去均值，未对方差进行处理。
            if useStd:
                t.div_(s)
        return img

    @classmethod
    def image_np2tensor(cls, imgNdarry):
        if imgNdarry.shape[0] != 1 and imgNdarry.shape[0] != 3:
            imgNdarry = np.transpose(imgNdarry, (2, 0, 1))  # H*W*C ==> C*H*W
        imgMap = torch.from_numpy(imgNdarry.astype(np.float32))
        if imgMap.max() > 1:
            imgMap /= 255
        return imgMap

    @classmethod
    def image_tensor2np(cls, imgMap):
        if not torch.is_tensor(imgMap): return None
        imgNdarry = imgMap.detach().cpu().numpy()
        if imgNdarry.shape[0] == 1 or imgNdarry.shape[0] == 3:
            imgNdarry = np.transpose(imgNdarry, (1, 2, 0))  # C*H*W ==> H*W*C
        return imgNdarry

    @classmethod
    def image_fliplr(cls, imgNdarry):
        if imgNdarry.ndim == 3:
            # np.fliplr 左右翻转
            imgNdarry = np.transpose(np.fliplr(np.transpose(imgNdarry, (0, 2, 1))), (0, 2, 1))
        elif imgNdarry.ndim == 4:
            for i in range(imgNdarry.shape[0]):
                imgNdarry[i] = np.transpose(np.fliplr(np.transpose(imgNdarry[i], (0, 2, 1))), (0, 2, 1))
        return imgNdarry.astype(float)

    @classmethod
    def center_calculate(cls, img, kps=None, cType="imgCenter"):
        h, w, _ = img.shape
        if cType == "imgCenter":
            return [int(w / 2), int(h / 2)]
        elif cType == "kpsCenter" and kps is not None:
            c, n = [0, 0], 0
            for kp in kps:
                if kp[2] == 0: continue
                c[0] += kp[0]
                c[1] += kp[1]
                n += 1
            return [int(c[0] / n), int(c[1] / n)]

    @classmethod
    def kps_fliplr(cls, kpsMap, imgWidth):
        # 对坐标值的修改（Flip horizontal）
        kpsMap[:, 0] = imgWidth - kpsMap[:, 0]
        return kpsMap

    @classmethod
    def kps_heatmap(cls, kpsMap, imgShape, inpRes, outRes, kernelSize=3.0, sigma=1.0):
        _, h, w = imgShape  # C*H*W
        stride = inpRes / outRes
        sizeH, sizeW = int(h / stride), int(w / stride)  # 计算HeatMap尺寸
        kpsCount = len(kpsMap)
        sigma *= kernelSize
        # 将HeatMap大小设置网络最小分辨率
        heatmap = np.zeros((sizeH, sizeW, kpsCount), dtype=np.float32)
        for kIdx in range(kpsCount):
            # 检查高斯函数的任意部分是否在范围内
            kp_int = kpsMap[kIdx].to(torch.int32)
            ul = [int(kp_int[0] - sigma), int(kp_int[1] - sigma)]
            br = [int(kp_int[0] + sigma + 1), int(kp_int[1] + sigma + 1)]
            vis = 0 if (br[0] >= w or br[1] >= h or ul[0] < 0 or ul[1] < 0) else 1
            kpsMap[kIdx][2] *= vis

            # 将keypoints转化至指定分辨率下
            x = int(kpsMap[kIdx][0]) * 1.0 / stride
            y = int(kpsMap[kIdx][1]) * 1.0 / stride
            kernel = cls.heatmap_gaussian(sizeH, sizeW, center=[x, y], sigma=sigma)
            # 边缘修正
            kernel[kernel > 1] = 1
            kernel[kernel < 0.01] = 0
            heatmap[:, :, kIdx] = kernel
        heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1)))
        return heatmap.float(), kpsMap

    @classmethod
    def kps_heatmap_mulKps(cls, kpsMapArray, imgShape, inpRes, outRes, kernelSize=3.0, sigma=1.0):
        _, h, w = imgShape  # C*H*W
        stride = inpRes / outRes
        sizeH, sizeW = int(h / stride), int(w / stride)  # 计算HeatMap尺寸
        sigma *= kernelSize
        heatmapArray, kpsMapArray_new = [], []
        for kpsMap in kpsMapArray:
            kpsCount = len(kpsMap)
            # 将HeatMap大小设置网络最小分辨率
            heatmap = np.zeros((sizeH, sizeW, kpsCount), dtype=np.float32)
            for kIdx in range(kpsCount):
                # 检查高斯函数的任意部分是否在范围内
                kp_int = kpsMap[kIdx].to(torch.int32)
                ul = [int(kp_int[0] - sigma), int(kp_int[1] - sigma)]
                br = [int(kp_int[0] + sigma + 1), int(kp_int[1] + sigma + 1)]
                vis = 0 if (br[0] >= w or br[1] >= h or ul[0] < 0 or ul[1] < 0) else 1
                kpsMap[kIdx][2] *= vis

                # 将keypoints转化至指定分辨率下
                x = int(kpsMap[kIdx][0]) * 1.0 / stride
                y = int(kpsMap[kIdx][1]) * 1.0 / stride
                kernel = cls.heatmap_gaussian(sizeH, sizeW, center=[x, y], sigma=sigma)
                # 边缘修正
                kernel[kernel > 1] = 1
                kernel[kernel < 0.01] = 0
                heatmap[:, :, kIdx] = kernel
            heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1)))
            heatmapArray.append(heatmap.float())
            kpsMapArray_new.append(kpsMap)
        return heatmapArray, kpsMapArray_new

    @classmethod
    def kps_fromHeatmap(cls, heatmap, cenMap, scale, res, mode="batch"):
        if mode == "single":
            return final_preds(heatmap.unsqueeze(0), cenMap.unsqueeze(0), scale.unsqueeze(0), res)[0]
        elif mode == "batch":
            preds = final_preds(heatmap, cenMap, scale, res)
            scores = torch.from_numpy(np.max(heatmap.detach().cpu().numpy(), axis=(2, 3)).astype(np.float32))
            return preds, scores

    @classmethod
    def kps_fromHeatmap_mul(cls, multiOuts, cenMap, scale, res):
        mc, bs, k, _, _ = multiOuts.shape
        predsMulti = torch.stack([final_preds(multiOuts[mcIdx], cenMap, scale, res) for mcIdx in range(mc)], 0)
        predsMean = torch.mean(predsMulti, dim=0)
        scoresMulti = torch.from_numpy(np.max(multiOuts.detach().cpu().numpy(), axis=(3, 4)).astype(np.float32))
        scoresMean = torch.mean(scoresMulti, dim=0)
        return predsMulti, predsMean, scoresMulti, scoresMean

    @classmethod
    def kps_fromHeatmap3(cls, heatmap, cenMap, scale, res):
        return final_preds(heatmap, cenMap, scale, res)

    @classmethod
    def kps_getLabeledCount(cls, kpsGate):
        return len([item for item in kpsGate.detach().reshape(-1).cpu().data.numpy() if item > 0])

    @classmethod
    def heatmap_gaussian(cls, h, w, center, sigma=3.0):
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        D2 = (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)

    @classmethod
    def draw_point(cls, img, coord, color=(0, 95, 191), radius=3, thickness=-1, text=None, textScale=1.0, textColor=(255, 255, 255)):
        img, x, y = img.astype(int), round(coord[0]), round(coord[1])
        if x > 1 and y > 1:
            cv2.circle(img, (x, y), color=color, radius=radius, thickness=thickness)
            if text is not None:
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, textScale, textColor, 2)
        return img

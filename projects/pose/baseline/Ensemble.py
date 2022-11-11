# -*- coding: utf-8 -*-
import numpy as np
import random
import argparse
import datetime
import torch
from torch.optim.adam import Adam as TorchAdam
from torch.utils.data.dataloader import DataLoader as TorchDataLoader

import GLOB as glob
import datasources
import datasets
import models

from utils.base.log import Logger
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc
from utils.evaluation import EvaluationUtils as eval
from utils.losses import GateJointMSELoss, AvgCounter, AvgCounters

import matplotlib
matplotlib.use('Agg')


def main(args):
    allTM = datetime.datetime.now()
    logger = glob.getValue("logger")
    logger.print("L1", "=> {}, start".format(args.experiment))
    args.best_acc = -1.
    args.best_epoch = 0

    # region 1. dataloader initialize
    loadingTM = datetime.datetime.now()
    # data loading
    dataSource = datasources.__dict__[args.dataSource]()
    semiTrainData, validData, labeledData, unlabeledData, means, stds = dataSource.getSemiData(args.trainCount, args.validCount, args.labelRatio)
    args.kpsCount, args.imgType, args.pck_ref = dataSource.kpsCount, dataSource.imgType, dataSource.pck_ref
    # train-set dataloader
    trainDS = datasets.__dict__["DS"]("train", labeledData, means, stds, isAug=True, isDraw=False, **vars(args))
    trainLoader = TorchDataLoader(trainDS, batch_size=args.trainBS, shuffle=False, pin_memory=True, drop_last=False)
    # valid-set dataloader
    validDS = datasets.__dict__["DS"]("valid", validData, means, stds, isAug=False, isDraw=False, **vars(args))
    validLoader = TorchDataLoader(validDS, batch_size=args.inferBS, shuffle=False, pin_memory=True, drop_last=False)
    logger.print("L1", "=> initialized {} Dataset loaders".format(args.dataSource), start=loadingTM)
    # endregion

    # region 2. modules initialize
    loadingTM = datetime.datetime.now()
    # Mean-Teacher Module 1
    model = models.__dict__["HG"](args.kpsCount, args.nStack).to(args.device)
    model_ens2 = models.__dict__["HG"](args.kpsCount, args.nStack).to(args.device)
    model_ens3 = models.__dict__["HG"](args.kpsCount, args.nStack).to(args.device)
    optim = TorchAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optim_ens2 = TorchAdam(model_ens2.parameters(), lr=args.lr, weight_decay=args.wd)
    optim_ens3 = TorchAdam(model_ens3.parameters(), lr=args.lr, weight_decay=args.wd)
    hg_pNum = sum(p.numel() for p in model.parameters())
    logc = "=> initialized HG models (nStack: {}, params: {})".format(args.nStack, format(hg_pNum / 1000000.0, ".2f"))
    logger.print("L1", logc, start=loadingTM)
    # endregion

    # region 3. iteration
    logger.print("L1", "=> start, Hourglass with supervised learning")
    for epo in range(args.epochs):
        epoTM = datetime.datetime.now()

        # region 3.1 model training and validating
        startTM = datetime.datetime.now()
        pec_loss = train(trainLoader, model, optim, args)
        pec2_loss = train(trainLoader, model_ens2, optim_ens2, args)
        pec3_loss = train(trainLoader, model_ens3, optim_ens3, args)
        logger.print("L3", "model training finished...", start=startTM)
        startTM = datetime.datetime.now()
        preds1Array, preds2Array, preds3Array, preds_mean_Array, errs1Array, accs1Array, errs2Array, accs2Array, errs3Array, accs3Array, errs_mean_Array, accs_mean_Array = validate(validLoader, model, model_ens2, model_ens3, args)
        logger.print("L3", "model validating finished...", start=startTM)
        # endregion

        # region 3.2 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = accs_mean_Array[-1][-1] > args.best_acc
        if is_best:
            args.best_epoch = epo + 1
            args.best_acc = accs_mean_Array[-1][-1]
        # model storage
        comm.ckpt_save({
            'epoch': epo + 1,
            'model': "HG",
            'best_acc': args.best_acc,
            'best_epoch': args.best_epoch,
            'state_dict': model.state_dict(),
            'optim': optim.state_dict()
        }, is_best, ckptPath="{}/ckpts/model".format(args.basePath))
        logger.print("L3", "model storage finished...", start=startTM)
        # endregion

        # region 3.3 log storage
        startTM = datetime.datetime.now()
        log_dataItem = {"pec_loss": pec_loss, "errs1Array": errs1Array, "accs1Array": accs1Array, "errs2Array": errs2Array, "accs2Array": accs2Array,
                        "errs3Array": errs3Array, "accs3Array": accs3Array, "errs_mean_Array": errs_mean_Array, "accs_mean_Array": accs_mean_Array,
                        "preds1Array": preds1Array, "preds2Array": preds2Array, "preds3Array": preds3Array, "preds_mean_Array": preds_mean_Array}
        comm.json_save(log_dataItem, "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)
        if epo == 0:
            save_args = vars(args).copy()
            save_args.pop("device")
            comm.json_save(save_args, "{}/logs/args.json".format(args.basePath), isCover=True)
        logger.print("L3", "log storage finished...", start=startTM)
        # endregion

        # region 3.4 output result
        time_interval = logger._interval_format(seconds=(datetime.datetime.now() - epoTM).seconds*(args.epochs - (epo+1)))
        fmtc = "[{}/{} | remaining: {}] best acc: {} (epo: {}) | acc: {}, [{}]; err: {}, [{}] | pec_loss: {}, pec2_loss: {}, pec3_loss: {}"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), time_interval, format(args.best_acc, ".3f"), format(args.best_epoch, "3d"),
                           format(accs_mean_Array[-1][-1], ".3f"), setContent([accs1Array[-1][-1], accs2Array[-1][-1], accs3Array[-1][-1]], ".3f"),
                           format(errs_mean_Array[-1][-1], ".2f"), setContent([errs1Array[-1][-1], errs2Array[-1][-1], errs3Array[-1][-1]], ".3f"),
                           format(pec_loss, ".5f"), format(pec2_loss, ".5f"), format(pec3_loss, ".5f"))
        logger.print("L1", logc, start=epoTM)
        # endregion
    # endregion
    logger.print("L1", "[{}, All executing finished...]".format(args.experiment), start=allTM)


def train(trainLoader, model, optim, args):
    pec_counter = AvgCounter()
    pose_lossFunc = GateJointMSELoss(nStack=args.nStack, useKPsGate=True).to(args.device)
    model.train()
    for bat, (imgMap, kpsHeatmap, meta) in enumerate(trainLoader):
        optim.zero_grad()
        # region 1. data organize
        imgMap = torch.autograd.Variable(imgMap.to(args.device, non_blocking=True), requires_grad=True)
        kpsHeatmap = torch.autograd.Variable(kpsHeatmap.to(args.device, non_blocking=True), requires_grad=True)
        kpsGate = torch.autograd.Variable(meta['kpsWeight'].to(args.device, non_blocking=True), requires_grad=True)
        # endregion

        # region 2. model forward
        outs = model(imgMap)
        # endregion

        # region 3. pose estimation constraint
        pec_sum, pec_count = pose_lossFunc(outs, kpsHeatmap, kpsGate)
        pec_loss = args.poseWeight * ((pec_sum / pec_count) if pec_count > 0 else pec_sum)
        pec_counter.update(pec_loss.item(), pec_count)
        # endregion

        # region 4. calculate total loss & update model
        total_loss = pec_loss
        total_loss.backward()  # retain_graph=True
        optim.step()
        # endregion
    return pec_counter.avg


def validate(validLoader, model, model2, model3, args):
    errs_countersArray, accs_countersArray = [AvgCounters() for item in range(args.nStack)], [AvgCounters() for item in range(args.nStack)]
    errs2_countersArray, accs2_countersArray = [AvgCounters() for item in range(args.nStack)], [AvgCounters() for item in range(args.nStack)]
    errs3_countersArray, accs3_countersArray = [AvgCounters() for item in range(args.nStack)], [AvgCounters() for item in range(args.nStack)]
    errs_mean_countersArray, accs_mean_countersArray = [AvgCounters() for item in range(args.nStack)], [AvgCounters() for item in range(args.nStack)]
    preds1Array, preds2Array, preds3Array, preds_mean_Array = [], [], [], []
    model.eval()
    model2.eval()
    model3.eval()
    with torch.no_grad():
        for bat, (imgMap, kpsHeatmap, meta) in enumerate(validLoader):
            # region 1. data organize
            imgMap = imgMap.to(args.device, non_blocking=True)
            bs, k, _, _ = kpsHeatmap.shape
            # endregion

            # region 2. model forward
            outs = model(imgMap).cpu()  # 多模型预测结果 [bs, nstack, k, h, w]
            outs2 = model2(imgMap).cpu()
            outs3 = model3(imgMap).cpu()
            outs_mean = torch.mean(torch.stack([outs, outs2, outs3], 3), 3)
            for sIdx in range(args.nStack):
                # region outs
                preds = proc.kps_fromHeatmap3(outs[:, sIdx], meta['center'], meta['scale'], [args.outRes, args.outRes])
                # calculate the error and accuracy
                errs, accs = eval.accuracy(preds, meta['kpsMap'], args.pck_ref, args.pck_thr)
                for idx in range(k + 1):
                    errs_countersArray[sIdx].update(idx, errs[idx].item(), bs)
                    accs_countersArray[sIdx].update(idx, accs[idx].item(), bs)
                if sIdx == args.nStack-1:
                    for bIdx in range(bs):
                        for kIdx in range(k):
                            preds1Array.append({"kpID": "{}_{}".format(meta["imageID"][bIdx], kIdx), "coord": preds[bIdx, kIdx].cpu().numpy().tolist()})
                # endregion

                # region outs2
                preds2 = proc.kps_fromHeatmap3(outs2[:, sIdx], meta['center'], meta['scale'], [args.outRes, args.outRes])
                # calculate the error and accuracy
                errs2, accs2 = eval.accuracy(preds2, meta['kpsMap'], args.pck_ref, args.pck_thr)
                for idx in range(k + 1):
                    errs2_countersArray[sIdx].update(idx, errs2[idx].item(), bs)
                    accs2_countersArray[sIdx].update(idx, accs2[idx].item(), bs)
                if sIdx == args.nStack-1:
                    for bIdx in range(bs):
                        for kIdx in range(k):
                            preds2Array.append({"kpID": "{}_{}".format(meta["imageID"][bIdx], kIdx), "coord": preds2[bIdx, kIdx].cpu().numpy().tolist()})
                # endregion

                # region outs3
                preds3 = proc.kps_fromHeatmap3(outs3[:, sIdx], meta['center'], meta['scale'], [args.outRes, args.outRes])
                # calculate the error and accuracy
                errs3, accs3 = eval.accuracy(preds3, meta['kpsMap'], args.pck_ref, args.pck_thr)
                for idx in range(k + 1):
                    errs3_countersArray[sIdx].update(idx, errs3[idx].item(), bs)
                    accs3_countersArray[sIdx].update(idx, accs3[idx].item(), bs)
                if sIdx == args.nStack-1:
                    for bIdx in range(bs):
                        for kIdx in range(k):
                            preds3Array.append({"kpID": "{}_{}".format(meta["imageID"][bIdx], kIdx), "coord": preds3[bIdx, kIdx].cpu().numpy().tolist()})
                # endregion

                # region outs_mean
                preds_mean = proc.kps_fromHeatmap3(outs_mean[:, sIdx], meta['center'], meta['scale'], [args.outRes, args.outRes])
                # calculate the error and accuracy
                errs_mean, accs_mean = eval.accuracy(preds_mean, meta['kpsMap'], args.pck_ref, args.pck_thr)
                for idx in range(k + 1):
                    errs_mean_countersArray[sIdx].update(idx, errs_mean[idx].item(), bs)
                    accs_mean_countersArray[sIdx].update(idx, accs_mean[idx].item(), bs)
                if sIdx == args.nStack-1:
                    for bIdx in range(bs):
                        for kIdx in range(k):
                            preds_mean_Array.append({"kpID": "{}_{}".format(meta["imageID"][bIdx], kIdx), "coord": preds_mean[bIdx, kIdx].cpu().numpy().tolist()})
                # endregion
            # endregion
    return preds1Array, preds2Array, preds3Array, preds_mean_Array, [item.avg() for item in errs_countersArray], [item.avg() for item in accs_countersArray], [item.avg() for item in errs2_countersArray], [item.avg() for item in accs2_countersArray], [item.avg() for item in errs3_countersArray], [item.avg() for item in accs3_countersArray], [item.avg() for item in errs_mean_countersArray], [item.avg() for item in accs_mean_countersArray]


def setVariable(dataItem, deviceID):
    return torch.autograd.Variable(dataItem.to(deviceID, non_blocking=True), requires_grad=True)


def setContent(dataArray, fmt):
    strContent = ""
    for dataIdx, dataItem in enumerate(dataArray):
        if dataIdx == len(dataArray)-1:
            strContent += "{}".format(format(dataItem, fmt))
        else:
            strContent += "{}, ".format(format(dataItem, fmt))
    return strContent


def setArgs(args, params):
    dict_args = vars(args)
    if params is not None:
        for key in params.keys():
            if key in dict_args.keys():
                dict_args[key] = params[key]
    for key in dict_args.keys():
        if dict_args[key] == "True": dict_args[key] = True
        if dict_args[key] == "False": dict_args[key] = False
    return argparse.Namespace(**dict_args)


def exec(expMark="HG", params=None):
    random_seed = 1388
    random.seed(random_seed)
    np.random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = initArgs(params)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.experiment = "{}({}_{})_{}_{}".format(args.dataSource, args.trainCount, args.labelRatio, expMark, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    args.basePath = "{}/{}".format(glob.expr, args.experiment)
    glob.setValue("logger", Logger(args.experiment, consoleLevel="L1"))
    main(args)


def initArgs(params=None):
    # region 1. Parameters
    parser = argparse.ArgumentParser(description="Pose Estimation with SSL")
    # Dataset setting
    parser.add_argument("--dataSource", default="Pranav", choices=["Sniffing", "Pranav", "FLIC"])
    parser.add_argument("--trainCount", default=100, type=int)
    parser.add_argument("--validCount", default=500, type=int)
    parser.add_argument("--labelRatio", default=0.3, type=float)
    # Model structure
    parser.add_argument("--nStack", default=3, type=int, help="the number of stage in Multiple Pose Model")
    parser.add_argument("--inpRes", default=256, type=int, help="model input resolution (default: 256)")
    parser.add_argument("--outRes", default=64, type=int, help="model output resolution (default: 64)")
    # Training strategy
    parser.add_argument("--epochs", default=500, type=int, help="the number of total epochs")
    parser.add_argument("--trainBS", default=4, type=int, help="the batchSize of training")
    parser.add_argument("--inferBS", default=16, type=int, help="the batchSize of infering")
    parser.add_argument("--lr", default=2.5e-4, type=float, help="initial learning rate")
    parser.add_argument("--wd", default=0, type=float, help="weight decay (default: 0)")
    parser.add_argument("--power", default=0.9, type=float, help="power for learning rate decay")
    # Data augment
    parser.add_argument("--useFlip", default="True", help="whether add flip augment")
    parser.add_argument("--scaleRange", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rotRange", default=30.0, type=float, help="rotation factor")
    parser.add_argument("--useOcclusion", default="False", help="whether add occlusion augment")
    parser.add_argument("--numOccluder", default=8, type=int, help="number of occluder to add in")
    # Data augment (to teacher in Mean-Teacher)
    parser.add_argument("--scaleRange_ema", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rotRange_ema", default=30.0, type=float, help="rotation factor")
    parser.add_argument("--useOcclusion_ema", default="False", help="whether add occlusion augment")
    parser.add_argument("--numOccluder_ema", default=8, type=int, help="number of occluder to add in")
    # Hyper-parameter
    parser.add_argument("--poseWeight", default=10.0, type=float, help="the weight of pose loss (default: 10.0)")
    # misc
    parser.add_argument("--pck_thr", default=0.2, type=float)
    # endregion
    args = setArgs(parser.parse_args(), params)
    return args


if __name__ == "__main__":
    exec()

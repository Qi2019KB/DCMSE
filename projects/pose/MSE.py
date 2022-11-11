# -*- coding: utf-8 -*-
import argparse
import datetime
import random

import numpy as np
import torch
from torch.optim.adam import Adam as TorchAdam
from torch.utils.data.dataloader import DataLoader as TorchDataLoader

import GLOB as glob
import datasources
import datasets
import models

from utils.base.log import Logger
from utils.base.comm import CommUtils as comm
from utils.parameters import meanConsWeight_increase, stackConsWeight_increase
from utils.process import ProcessUtils as proc
from utils.evaluation import EvaluationUtils as eval
from utils.losses import GateJointMSELoss, GateJointDistLoss, AvgCounter, AvgCounters

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
    trainDS = datasets.__dict__["DS"]("train", semiTrainData, means, stds, isAug=True, isDraw=False, **vars(args))
    trainLoader = TorchDataLoader(trainDS, batch_size=args.trainBS, shuffle=False, pin_memory=True, drop_last=False)
    # valid-set dataloader
    validDS = datasets.__dict__["DS"]("valid", validData, means, stds, isAug=False, isDraw=False, **vars(args))
    validLoader = TorchDataLoader(validDS, batch_size=args.inferBS, shuffle=False, pin_memory=True, drop_last=False)
    logger.print("L1", "=> initialized {} Dataset loaders".format(args.dataSource), start=loadingTM)
    # endregion

    # region 2. modules initialize
    loadingTM = datetime.datetime.now()
    model = models.__dict__["HG_ms"](args.kpsCount, args.nStack, args.nStream).to(args.device)
    optim = TorchAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    hg_pNum = sum(p.numel() for p in model.parameters())
    logc = "=> initialized HG_ms models (nStack: {}, params: {})".format(args.nStack, format(hg_pNum / 1000000.0, ".2f"))
    logger.print("L1", logc, start=loadingTM)
    # endregion

    # region 3. iteration
    logger.print("L1", "=> start, Hourglass_ms with semi-supervised learning")
    for epo in range(args.epochs):
        epoTM = datetime.datetime.now()

        # region 3.1 update dynamic parameters
        args.meanConsWeight = meanConsWeight_increase(epo, args)
        args.stackConsWeight = stackConsWeight_increase(epo, args)
        # endregion

        # region 3.2 model training and validating
        startTM = datetime.datetime.now()
        pec_loss, mcc_loss, scc_loss = train(trainLoader, model, optim, args)
        logger.print("L3", "model training finished...", start=startTM)
        startTM = datetime.datetime.now()
        predsArray, errsArray_ms, accsArray_ms, errsArray, accsArray = validate(validLoader, model, args)
        logger.print("L3", "model validating finished...", start=startTM)
        # endregion

        # region 3.3 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = accsArray[-1][-1] > args.best_acc
        if is_best:
            args.best_epoch = epo + 1
            args.best_acc = accsArray[-1][-1]
        # model storage
        comm.ckpt_save({
            'epoch': epo + 1,
            'model': "HG_ms",
            'best_acc': args.best_acc,
            'best_epoch': args.best_epoch,
            'state_dict': model.state_dict(),
            'optim': optim.state_dict()
        }, is_best, ckptPath="{}/ckpts/model".format(args.basePath))
        logger.print("L3", "model storage finished...", start=startTM)
        # endregion

        # region 3.4 log storage
        startTM = datetime.datetime.now()
        log_dataItem = {"pose_loss": pec_loss, "mcc_loss": mcc_loss, "scc_loss": scc_loss,
                        "predsArray": predsArray, "errsArray": errsArray, "accsArray": accsArray, "errsArray_ms": errsArray_ms, "accsArray_ms": accsArray_ms}
        comm.json_save(log_dataItem, "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)
        if epo == 0:
            save_args = vars(args).copy()
            save_args.pop("device")
            comm.json_save(save_args, "{}/logs/args.json".format(args.basePath), isCover=True)
        logger.print("L3", "log storage finished...", start=startTM)
        # endregion

        # region 3.5 output result
        fmtc = "[{}/{} | meanW: {}, stackW: {}] best acc: {} (epo: {}) | acc: {}, err: {} | pose_loss: {}, mcc_loss: {}, scc_loss: {}"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(args.meanConsWeight, ".2f"), format(args.stackConsWeight, ".2f"),
                           format(args.best_acc, ".3f"), format(args.best_epoch, "3d"), format(accsArray[-1][-1], ".3f"), format(errsArray[-1][-1], ".2f"),
                           format(pec_loss, ".5f"), format(mcc_loss, ".5f"), format(scc_loss, ".5f"))
        logger.print("L1", logc)
        for i in range(args.nStack):
            fmtc = "[{}/{}] stack{}' err: [{}] | stacks' acc: [{}]"
            logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), i+1,
                               setContent(np.array(errsArray_ms)[i, :, -1].tolist(), ".2f"), setContent(np.array(accsArray_ms)[i, :, -1].tolist(), ".3f"))
            logger.print("L2", logc)
        time_interval = logger._interval_format(seconds=(datetime.datetime.now() - epoTM).seconds*(args.epochs - (epo+1)))
        fmtc = "[{}/{} | time remaining: {}] ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), time_interval)
        logger.print("L1", logc, start=epoTM)
        # endregion
    # endregion

    logger.print("L1", "[{}, All executing finished...]".format(args.experiment), start=allTM)


def train(trainLoader, model, optim, args):
    pec_counter, mcc_counter, scc_counter = AvgCounter(), AvgCounter(), AvgCounter()
    pose_lossFunc = GateJointMSELoss(nStack=args.nStack, useKPsGate=True, useSampleWeight=True).to(args.device)
    consistency_lossFunc = GateJointDistLoss().to(args.device)
    model.train()
    for bat, (imgMap, kpsHeatmap, meta) in enumerate(trainLoader):
        optim.zero_grad()
        # region 1. data organize
        sampleWeight = cal_sampleWeight(meta['islabeled'], args)
        imgMap = setVariable(imgMap, args.device)
        kpsHeatmap = setVariable(kpsHeatmap, args.device)
        kpsGate = setVariable(meta['kpsWeight'], args.device)
        bs = kpsHeatmap.size(0)
        sampleWeight_negative = setVariable(torch.where(sampleWeight == 1, torch.zeros_like(sampleWeight), torch.ones_like(sampleWeight)), args.device)
        # endregion

        # region 2. model forward
        outs_ms = model(imgMap)
        # endregion

        # region 3. pose estimation constraint
        # multiple stream predictions:
        pec_sum, pec_count = 0., 0
        for stIdx in range(args.nStream):
            loss, n = pose_lossFunc(outs_ms[:, :, stIdx], kpsHeatmap, kpsGate, sampleWeight)
            pec_sum += loss
            pec_count += n
        pec_loss = args.poseWeight * ((pec_sum / pec_count) if pec_count > 0 else pec_sum)
        pec_counter.update(pec_loss.item(), pec_count)
        # endregion

        # region 4. mean-multiple consistency constraint
        if args.meanConsWeight > 0:
            mcc_sum, mcc_count = 0., 0
            outs_ms_mean = torch.mean(outs_ms, 2).clone().detach_()
            for sIdx in range(args.nStack):
                for stIdx in range(args.nStream):
                    loss, n = consistency_lossFunc(outs_ms[:, sIdx, stIdx], outs_ms_mean[:, sIdx])
                    mcc_sum += loss
                    mcc_count += n
            mcc_loss = args.meanConsWeight * ((mcc_sum / mcc_count) if mcc_count > 0 else mcc_sum)
            mcc_counter.update(mcc_loss.item(), mcc_count)
        else:
            mcc_loss = 0
            mcc_counter.update(0., 1)
        # endregion

        # region 5. stack consistency constraint
        if args.stackConsWeight > 0:
            outs_b = outs_ms[:, 2].clone().detach_()
            scc_sum, scc_count = consistency_lossFunc(outs_ms[:, 0], outs_b)
            scc_loss = args.stackConsWeight * ((scc_sum / scc_count) if scc_count > 0 else scc_sum)
            scc_counter.update(scc_loss.item(), scc_count)
        else:
            scc_loss = 0
            scc_counter.update(0., 1)
        # endregion

        # region 6. calculate total loss & update model
        # cal total loss
        total_loss = pec_loss + mcc_loss + scc_loss
        # backward
        total_loss.backward()
        optim.step()
        # endregion
    return pec_counter.avg, mcc_counter.avg, scc_counter.avg


def validate(validLoader, model, args):
    errs_countersArray, accs_countersArray = [[AvgCounters() for j in range(args.nStream)] for i in range(args.nStack)], [[AvgCounters() for j in range(args.nStream)] for i in range(args.nStack)]
    errs_counters, accs_counters = [AvgCounters() for item in range(args.nStack)], [AvgCounters() for item in range(args.nStack)]
    stackPreds_array = [[[] for j in range(args.nStream)] for i in range(args.nStack)]
    model.eval()
    with torch.no_grad():
        for bat, (imgMap, kpsHeatmap, meta) in enumerate(validLoader):
            # region 1. data organize
            imgMap = imgMap.to(args.device, non_blocking=True)
            bs, k, _, _ = kpsHeatmap.shape
            # endregion

            # region 2. model forward
            outs_ms = model(imgMap)
            for sIdx in range(args.nStack):
                combined_ms_preds = []
                for stIdx in range(args.nStream):
                    preds = proc.kps_fromHeatmap3(outs_ms[:, sIdx, stIdx].cpu(), meta['center'], meta['scale'], [args.outRes, args.outRes])
                    stackPreds_array[sIdx][stIdx] += [item for item in preds.numpy().tolist()]
                    combined_ms_preds.append(preds)
                    # calculate the error and accuracy
                    errs, accs = eval.accuracy(preds, meta['kpsMap'], args.pck_ref, args.pck_thr)
                    for kIdx in range(k + 1):
                        errs_countersArray[sIdx][stIdx].update(kIdx, errs[kIdx].item(), bs)
                        accs_countersArray[sIdx][stIdx].update(kIdx, accs[kIdx].item(), bs)
                combined_ms_preds = torch.stack(combined_ms_preds, 1)  # [bs, nStream, k, h, w]
                mean_ms_preds = torch.mean(combined_ms_preds, 1)
                ms_errs, ms_accs = eval.accuracy(mean_ms_preds, meta['kpsMap'], args.pck_ref, args.pck_thr)
                for kIdx in range(k + 1):
                    errs_counters[sIdx].update(kIdx, ms_errs[kIdx].item(), bs)
                    accs_counters[sIdx].update(kIdx, ms_accs[kIdx].item(), bs)
            # endregion

    return stackPreds_array, [[errs_countersArray[i][j].avg() for j in range(args.nStream)] for i in range(args.nStack)], [[accs_countersArray[i][j].avg() for j in range(args.nStream)] for i in range(args.nStack)], [item.avg() for item in errs_counters], [item.avg() for item in accs_counters]


def cal_sampleWeight(islabeled, args):
    # calculate the weight of the gt-sample (w = 1.0) and pseudo-label (w = args.cur_pw)
    islabeled = islabeled.to(args.device, non_blocking=True)
    sampleWeight = islabeled.detach().float()
    sampleWeight_pseudo = 0. * torch.ones_like(sampleWeight)
    sampleWeight = setVariable(torch.where(islabeled > 0, sampleWeight, sampleWeight_pseudo), args.device).unsqueeze(-1)
    return sampleWeight


def setContent(dataArray, fmt):
    strContent = ""
    for dataIdx, dataItem in enumerate(dataArray):
        if dataIdx == len(dataArray)-1:
            strContent += "{}".format(format(dataItem, fmt))
        else:
            strContent += "{}, ".format(format(dataItem, fmt))
    return strContent


def setVariable(dataItem, deviceID):
    return torch.autograd.Variable(dataItem.to(deviceID, non_blocking=True), requires_grad=True)


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


def exec(expMark, params=None):
    random_seed = 1388
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

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
    parser.add_argument("--nStream", default=3, type=int, help="the number of stage in Multiple Stream")
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
    parser.add_argument("--poseWeight", default=10.0, type=float)
    parser.add_argument("--meanConsWeight_max", default= 0.0, type=float)  # 50.0
    parser.add_argument("--meanConsWeight_rampup", default=50, type=int)
    parser.add_argument("--stackConsWeight_max", default=0.0, type=float)  # 5.0
    parser.add_argument("--stackConsWeight_rampup", default=50, type=int)
    # misc
    parser.add_argument("--pck_thr", default=0.2, type=float)
    # endregion
    args = setArgs(parser.parse_args(), params)
    return args


if __name__ == "__main__":
    exec()

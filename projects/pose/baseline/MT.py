# -*- coding: utf-8 -*-
import os
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
from utils.parameters import consWeight_increase, update_ema_variables
from utils.process import ProcessUtils as proc
from utils.augment import AugmentUtils as aug
from utils.evaluation import EvaluationUtils as eval
from utils.losses import GateJointMSELoss, GateJointDistLoss, AvgCounter, AvgCounters

import matplotlib
matplotlib.use('Agg')


def main(args):
    allTM = datetime.datetime.now()
    logger = glob.getValue("logger")
    logger.print("L1", "=> {}, start".format(args.experiment))
    args.global_step = 0
    args.best_acc = -1.
    args.best_epoch = 0

    # region 1. dataloader initialize
    loadingTM = datetime.datetime.now()
    # data loading
    dataSource = datasources.__dict__[args.dataSource]()
    semiTrainData, validData, labeledData, unlabeledData, means, stds = dataSource.getSemiData(args.trainCount, args.validCount, args.labelRatio)
    args.kpsCount, args.imgPath, args.imgType, args.pck_ref = dataSource.kpsCount, dataSource.imgPath, dataSource.imgType, dataSource.pck_ref
    # train-set dataloader
    trainDS = datasets.__dict__["DS_mt"]("train", semiTrainData, means, stds, isAug=True, isDraw=False, **vars(args))
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
    model_ema = models.__dict__["HG"](args.kpsCount, args.nStack, nograd=True).to(args.device)
    optim = TorchAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    hg_pNum = sum(p.numel() for p in model.parameters())
    logc = "=> initialized MT models (nStack: {}, params: {})".format(args.nStack, format(hg_pNum / 1000000.0, ".2f"))
    logger.print("L1", logc, start=loadingTM)
    # endregion

    # region 3. iteration
    logger.print("L1", "=> start, Mean Teacher with semi-supervised learning")
    for epo in range(args.epochs):
        epoTM = datetime.datetime.now()

        # region 3.1 update dynamic parameters
        args.consWeight = consWeight_increase(epo, args)
        # endregion

        # region 3.3 model training and validating
        startTM = datetime.datetime.now()
        ecc_loss, pec_loss = train(trainLoader, model, model_ema, optim, args)
        logger.print("L3", "model training finished...", start=startTM)
        startTM = datetime.datetime.now()
        errsArray, accsArray = validate(validLoader, model_ema, args)
        logger.print("L3", "model validating finished...", start=startTM)
        # endregion

        # region 3.4 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = accsArray[-1][-1] > args.best_acc
        if is_best:
            args.best_epoch = epo + 1
            args.best_acc = accsArray[-1][-1]
        # model storage
        comm.ckpt_save({
            'epoch': epo + 1,
            'model': "HG",
            'global_step': args.global_step,
            'best_acc': args.best_acc,
            'best_epoch': args.best_epoch,
            'state_dict': model.state_dict(),
            'state_dict_ema': model.state_dict(),
            'optim': optim.state_dict()
        }, is_best, ckptPath="{}/ckpts/model".format(args.basePath))
        # endregion
        logger.print("L3", "model storage finished...", start=startTM)
        # endregion

        # region 3.3 log storage
        startTM = datetime.datetime.now()
        log_dataItem = {"pec_loss": pec_loss, "ecc_loss": ecc_loss, "errsArray": errsArray, "accsArray": accsArray}
        comm.json_save(log_dataItem, "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)
        if epo == 0:
            save_args = vars(args).copy()
            save_args.pop("device")
            comm.json_save(save_args, "{}/logs/args.json".format(args.basePath), isCover=True)
        logger.print("L3", "log storage finished...", start=startTM)
        # endregion

        # region 3.7 output result
        time_interval = logger._interval_format(seconds=(datetime.datetime.now() - epoTM).seconds*(args.epochs - (epo+1)))
        fmtc = "[{}/{} | remaining: {}] best acc: {} (epo: {}) | acc: {}, err: {} | pec_loss: {}, ecc_loss: {}"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), time_interval, format(args.best_acc, ".3f"), format(args.best_epoch, "3d"),
                           format(accsArray[-1][-1], ".3f"), format(errsArray[-1][-1], ".2f"), format(pec_loss, ".5f"), format(ecc_loss, ".5f"))
        logger.print("L1", logc, start=epoTM)
        # endregion
    # endregion

    logger.print("L1", "[{}, All executing finished...]".format(args.experiment), start=allTM)


def train(trainLoader, model, model_ema, optim, args):
    ecc_counter, pec_counter = AvgCounter(), AvgCounter()
    pose_lossFunc = GateJointMSELoss(nStack=args.nStack, useKPsGate=True, useSampleWeight=True).to(args.device)
    consistency_lossFunc = GateJointDistLoss().to(args.device)
    model.train()
    model_ema.train()
    for bat, (imgMap, kpsHeatmap, imgMap_ema, meta) in enumerate(trainLoader):
        optim.zero_grad()
        # region 1. data organize
        sampleWeight = calSampleWeight_train(meta['islabeled'], args)  # [bs]; calculate the weight of the gt-sample (w = 1.0) and pseudo-label (w = args.cur_pw)
        kpsGate = setVariable(meta['kpsWeight'], args.device)   # [bs, 9]
        kpsHeatmap = setVariable(kpsHeatmap, args.device)  # [bs, 9, 256, 256]

        imgMap = setVariable(imgMap, args.device)  # [bs, 3, 256, 256]
        warpmat = setVariable(meta['warpmat'], args.device)  # [bs, 2, 3]
        isflip = meta['isflip']  # [bs]

        imgMap_ema = setVariable(imgMap_ema, args.device)
        warpmat_ema = setVariable(meta['warpmat_ema'], args.device)  # [bs, 2, 3]
        isflip_ema = meta['isflip_ema']  # [bs]
        # endregion

        # region 2. model forward
        outs = model(imgMap)
        outs_ema = model_ema(imgMap_ema)
        # endregion

        # region 3. external consistency constraint (mean-teacher consistency)
        # calculate ecc_loss
        if args.usePEC:
            preds_v1 = aug.affine_back(outs[:, -1], warpmat, isflip).to(args.device, non_blocking=True)
            preds_v2 = aug.affine_back(outs_ema[:, -1], warpmat_ema, isflip_ema).to(args.device, non_blocking=True)
            ecc_sum, ecc_count = consistency_lossFunc(preds_v1, preds_v2)
            ecc_loss = args.consWeight * ((ecc_sum / ecc_count) if ecc_count > 0 else ecc_sum)
            ecc_counter.update(ecc_loss.item(), ecc_count)
        else:
            ecc_loss = 0.0
            ecc_counter.update(ecc_loss, 1)
        # endregion

        # region 4. pose estimation constraint
        pec_sum, pec_count = pose_lossFunc(outs, kpsHeatmap, kpsGate, sampleWeight)
        # cal & record the pec_loss
        pec_loss = args.poseWeight * ((pec_sum / pec_count) if pec_count > 0 else pec_sum)
        pec_counter.update(pec_loss.item(), pec_count)
        # endregion

        # region 6. calculate total loss & update model
        # cal total loss
        total_loss = ecc_loss + pec_loss
        # backward
        total_loss.backward()  # retain_graph=True
        optim.step()
        update_ema_variables(model, model_ema, args)  # update teacher by EMA
        # endregion
    return ecc_counter.avg, pec_counter.avg


def validate(validLoader, model, args):
    errs_countersArray, accs_countersArray = [AvgCounters() for item in range(args.nStack)], [AvgCounters() for item in range(args.nStack)]
    model.eval()
    with torch.no_grad():
        for bat, (imgMap, kpsHeatmap, meta) in enumerate(validLoader):
            # region 1. data organize
            imgMap = imgMap.to(args.device, non_blocking=True)
            bs, k, _, _ = kpsHeatmap.shape
            # endregion

            # region 2. model forward
            for sIdx in range(args.nStack):
                outs = model(imgMap)[:, sIdx].cpu()  # 多模型预测结果 [bs, nstack, k, h, w]
                preds, _ = proc.kps_fromHeatmap(outs, meta['center'], meta['scale'], [args.outRes, args.outRes])  # kps_fromHeatmap(heatmap, cenMap, scale, res)
                # calculate the error and accuracy
                errs, accs = eval.accuracy(preds, meta['kpsMap'], args.pck_ref, args.pck_thr)
                for idx in range(k + 1):
                    errs_countersArray[sIdx].update(idx, errs[idx].item(), bs)
                    accs_countersArray[sIdx].update(idx, accs[idx].item(), bs)
            # endregion

            # region 5. clearing the GPU Cache
            del imgMap, outs, preds, errs, accs, _
            # endregion
    return [item.avg() for item in errs_countersArray], [item.avg() for item in accs_countersArray]


def calSampleWeight_train(islabeled, args):
    # calculate the weight of the gt-sample (w = 1.0) and pseudo-label (w = args.cur_pw)
    islabeled = islabeled.to(args.device, non_blocking=True)  # [bs]
    sampleWeight = islabeled.detach().float()  # [bs]
    sampleWeight_pseudo = 0. * torch.ones_like(sampleWeight)  # [bs]
    sampleWeight = setVariable(torch.where(islabeled > 0, sampleWeight, sampleWeight_pseudo), args.device).unsqueeze(-1)
    return sampleWeight


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


def exec(expMark="MT", params=None):
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
    parser.add_argument("--usePEC", default="True")
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
    parser.add_argument("--isAug_ema", default="True")
    parser.add_argument("--scaleRange_ema", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rotRange_ema", default=30.0, type=float, help="rotation factor")
    parser.add_argument("--useOcclusion_ema", default="False", help="whether add occlusion augment")
    parser.add_argument("--numOccluder_ema", default=8, type=int, help="number of occluder to add in")
    # Hyper-parameter
    parser.add_argument("--poseWeight", default=10.0, type=float, help="the weight of pose loss (default: 10.0)")
    parser.add_argument("--consWeight_max", default=20.0, type=float, help="the max weight of consistency loss")
    parser.add_argument("--consWeight_rampup", default=50, type=int, help="length of the consistency loss ramp-up")
    parser.add_argument("--consWeight_start", default=0, type=int)
    # mean-teacher
    parser.add_argument("--ema_decay", default=0.999, type=float, help="ema variable decay rate (default: 0.999)")
    # misc
    parser.add_argument("--pck_thr", default=0.2, type=float)
    # endregion
    args = setArgs(parser.parse_args(), params)
    return args


if __name__ == "__main__":
    exec()

# -*- coding: utf-8 -*-
from projects.pose.baseline.HG import exec as HG
from projects.pose.baseline.Ensemble import exec as Ensemble
from projects.pose.baseline.MT import exec as MT
from projects.pose.baseline.ESCP import exec as ESCP
from projects.pose.baseline.DNCL import exec as DNCL

from projects.pose.MSE import exec as MSE
from projects.pose.DCMSE import exec as DCMSE


def exec():
    DCMSE("DCMSE", {"dataSource": "Pranav", "trainCount": 100, "validCount": 500, "labelRatio": 0.3, "epochs": 500})
    MSE("MSE", {"dataSource": "FLIC", "trainCount": 100, "validCount": 500, "labelRatio": 0.3, "epochs": 500})

    # region Baselines
    HG("HG", {"dataSource": "Sniffing", "trainCount": 100, "validCount": 500, "labelRatio": 0.3, "epochs": 500})
    Ensemble("Ensemble", {"trainCount": 100, "validCount": 500, "labelRatio": 0.3, "epochs": 500})
    DNCL("DNCL", {"trainCount": 100, "validCount": 500, "labelRatio": 0.3, "epochs": 500})
    MT("MT", {"trainCount": 100, "validCount": 500, "labelRatio": 0.3, "epochs": 500})
    ESCP("ESCP", {"trainCount": 100, "validCount": 500, "labelRatio": 0.3, "epochs": 500})
    # endregion


if __name__ == "__main__":
    exec()
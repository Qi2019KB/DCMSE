import numpy as np


def update_ema_variables(model, ema_model, args):
    global_step = args.global_step + 1
    alpha = min(1 - 1 / (global_step + 1), args.ema_decay)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    return global_step


# min --> max
def consWeight_increase(epo, args):
    return args.consWeight_max * _sigmoid_rampup(epo, args.consWeight_rampup)


# min --> max
def meanConsWeight_increase(epo, args):
    return args.meanConsWeight_max * _sigmoid_rampup(epo, args.meanConsWeight_rampup)


# min --> max
def stackConsWeight_increase(epo, args):
    return args.stackConsWeight_max * _sigmoid_rampup(epo, args.stackConsWeight_rampup)


def _sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
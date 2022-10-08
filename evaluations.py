import numpy as np


def RRSE(v, v_):
    '''
    RRSE
    v: real values
    v_: predict values
    '''
    n = np.sqrt (np.sum ((v - v_) ** 2))
    d = np.sqrt (np.sum ((v - np.mean (v_)) ** 2))
    return n / d


def CORR(v, v_):
    vm = v.mean (axis=0)
    v_m = v_.mean (axis=0)
    n = np.sum ((v - vm) * (v_ - v_m), axis=0)
    d = np.sqrt (np.sum ((v - vm) ** 2, axis=0) * np.sum ((v_ - v_m) ** 2, axis=0)) + 1e-5
    co = np.mean (n / d)
    return co


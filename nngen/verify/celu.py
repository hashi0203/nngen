from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def celu(features,
         lut_addrwidth=8, lut_clip=6.0,
         dtype=None, name=None, par=1,
         features_dtype=None, features_scale=1, features_shamt=0):

    features_point = 0 if features_dtype is None else features_dtype.point
    out_point = 0 if dtype is None else dtype.point
    out_shift = out_point - features_point

    mul = features * features_scale
    sra = mul >> features_shamt

    if dtype is None:
        raise ValueError('celu requires dtype to determine the value range.')

    out_width = dtype.width
    out_point = dtype.point
    out_scale = 2 ** (out_width - 1)

    def _celu_n(x):
        return np.around((np.exp(-x) - 1) * out_scale).astype(np.int64)

    addr_scale = lut_clip / (2 ** lut_addrwidth)
    lut = _celu_n(np.abs(sra) * addr_scale)

    p_th = 2 ** lut_addrwidth - 1
    n_th = -1 * p_th

    if out_point == 0:
        th_scale = out_scale
    elif out_point > 0:
        th_scale = out_scale >> out_point
    else:
        th_scale = out_scale << (-1 * out_point)

    n = np.where(sra < n_th, -th_scale, lut)
    out = np.where(sra >= 0, features, n)

    return out

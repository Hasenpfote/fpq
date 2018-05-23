#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def remap(value, src_min, src_max, dst_min, dst_max):
    '''Maps values from [`src_min`, `src_max`]  to [`dst_min`, `dst_max`].

    Args:
        value: The incoming value to be converted.
        src_min: Lower bound of the value current range.
        src_max: Upper bound of the value current range.
        dst_min: Lower bound of the value target range.
        dst_max: Upper bound of the value target range.

    Returns:
        The resulting value.

    Examples:
    '''
    return (((value - src_min) / (src_max - src_min)) * (dst_max - dst_min)) + dst_min

def _can_express_norm(nbits, dtype):
    '''Can express normalized integers?'''
    return nbits <= (1 + np.finfo(dtype).nmant)

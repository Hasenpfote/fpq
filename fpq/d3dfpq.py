#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def encode_fp_to_unorm(src, dst, *, nbits):
    '''Encode floating-points to unsigned normalized integers.

    Args:
        src: The type should be `np.float`, or an array in `np.float`.
        dst: The type should be `np.uint`.
        nbits: The number of `dst` bits to use.

    Returns:
        The resulting unsigned normalized integers.
    '''
    assert (0 < nbits <= (dst.itemsize * 8)), '`nbits` value is out of range.'
    return dst.dtype.type(np.around(src * src.dtype.type((1 << nbits) - 1)))

def decode_fp_from_unorm(src, dst, *, nbits):
    '''Decode floating-points from unsigned normalized integers.

    Args:
        src: The type should be `np.uint`, or an array in `np.uint`.
        dst: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.
    '''
    assert (0 < nbits <= (src.itemsize * 8)), '`nbits` value is out of range.'
    return dst.dtype.type(src) / dst.dtype.type((1 << nbits) - 1)

def encode_fp_to_snorm(src, dst, *, nbits):
    '''Encode floating-points to signed normalized integers.

    Args:
        src: The type should be `np.float`, or an array in `np.float`.
        dst: The type should be `np.uint`.
        nbits: The number of `dst` bits to use.

    Returns:
        The resulting unsigned normalized integers.
    '''
    assert (0 < nbits <= (dst.itemsize * 8)), '`nbits` value is out of range.'
    mask = np.invert(dst.dtype.type(np.iinfo(dst.dtype).max << nbits))
    sign = np.signbit(src)
    temp = dst.dtype.type(np.around(np.fabs(src) * src.dtype.type((1 << (nbits - 1)) - 1)))
    temp = (temp * np.logical_not(sign)) | ((np.invert(temp) & mask) * sign)
    return (sign << dst.dtype.type(nbits - 1)) | temp

def decode_fp_from_snorm(src, dst, *, nbits):
    '''Decode floating-points from signed normalized integers.

    Args:
        src: The type should be `np.uint`, or an array in `np.uint`.
        dst: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.
    '''
    assert (0 < nbits <= (src.itemsize * 8)), '`nbits` value is out of range.'
    mask = np.invert(src.dtype.type(np.iinfo(src.dtype).max << nbits))
    sign = (src >> src.dtype.type(nbits - 1)).astype(np.bool)
    temp = dst.dtype.type((src * np.logical_not(sign)) | ((np.invert(src) + src.dtype.type(1)) & mask) * sign)
    temp *= (sign * dst.dtype.type(-2.) + dst.dtype.type(1.))
    temp /= dst.dtype.type((1 << (nbits - 1)) - 1)
    return np.clip(temp, -1., 1.)

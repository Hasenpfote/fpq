#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def encode_fp_to_unorm(x, *, dtype=np.uint8, nbits=None):
    '''Encode floating-points to unsigned normalized integers.

    Args:
        x: The type should be `np.float`, or an array in `np.float`.
        dtype: The type should be `np.uint`.
        nbits: The number of `dst` bits to use.

    Returns:
        The resulting unsigned normalized integers.
    '''
    assert (x.dtype.kind == 'f'), '`dtype` of the argument `x` must be floating point types.'
    assert (dtype().dtype.kind == 'u'), '`dtype` of the argument `dtype` must be unsigned integer types.'
    max_nbits = dtype().itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    return dtype(np.around(x * x.dtype.type((1 << nbits) - 1)))

def decode_unorm_to_fp(x, *, dtype=np.float32, nbits=None):
    '''Decode unsigned normalized integers to floating-points.

    Args:
        x: The type should be `np.uint`, or an array in `np.uint`.
        dtype: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.
    '''
    assert (x.dtype.kind == 'u'), '`dtype` of the argument `x` must be unsigned integer types.'
    assert (dtype().dtype.kind == 'f'), '`dtype` of the argument `dtype` must be floating point types.'
    max_nbits = x.itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    return dtype(x) / dtype((1 << nbits) - 1)

def encode_fp_to_snorm(x, *, dtype=np.uint8, nbits=None):
    '''Encode floating-points to signed normalized integers.

    Args:
        x: The type should be `np.float`, or an array in `np.float`.
        dtype: The type should be `np.uint`.
        nbits: The number of `dst` bits to use.

    Returns:
        The resulting unsigned normalized integers.
    '''
    assert (x.dtype.kind == 'f'), '`dtype` of the argument `x` must be floating point types.'
    assert (dtype().dtype.kind == 'u'), '`dtype` of the argument `dtype` must be unsigned integer types.'
    max_nbits = dtype().itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    mask = np.invert(dtype(np.iinfo(nbits).max) << dtype(nbits))
    return dtype(np.around(x * x.dtype.type((1 << (nbits-1)) - 1))) & mask

def decode_snorm_to_fp(x, *, dtype=np.float32, nbits=None):
    '''Decode signed normalized integers to floating-points.

    Args:
        x: The type should be `np.uint`, or an array in `np.uint`.
        dtype: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.
    '''
    assert (x.dtype.kind == 'u'), '`dtype` of the argument `x` must be unsigned integer types.'
    assert (dtype().dtype.kind == 'f'), '`dtype` of the argument `dtype` must be floating point types.'
    max_nbits = x.itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    sign = x >> x.dtype.type(nbits-1)
    mask = x.dtype.type(np.iinfo(x.dtype).max) << x.dtype.type(nbits)
    temp = x | (sign * mask)
    temp = dtype(np.dtype(x.dtype.name[1:]).type(temp)) / dtype((1 << (nbits-1)) - 1)
    return np.maximum(temp, dtype(-1.))

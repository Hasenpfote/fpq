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

    Examples:
        >>> fp = np.float32(1.)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)

        >>> fp = np.array([0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)

        >>> fp = np.array([[0., 0.5, 1.], [1., 0.5, 0.]], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)
   '''
    max_nbits = dtype().itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    return dtype(np.around(x * x.dtype.type((1 << nbits) - 1)))

def decode_fp_from_unorm(x, *, dtype=np.float32, nbits=None):
    '''Decode floating-points from unsigned normalized integers.

    Args:
        x: The type should be `np.uint`, or an array in `np.uint`.
        dtype: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.

    Examples:
        >>> fp = np.float32(1.)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_fp_from_unorm(enc, dtype=np.float32, nbits=16)

        >>> fp = np.array([0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_fp_from_unorm(enc, dtype=np.float32, nbits=16)

        >>> fp = np.array([[0., 0.5, 1.], [1., 0.5, 0.]], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_fp_from_unorm(enc, dtype=np.float32, nbits=16)
    '''
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

    Examples:
        >>> fp = np.float32(-1.)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)

        >>> fp = np.array([-1.0, -0.5, 0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)

        >>> fp = np.array([[-1.0, -0.5, 0., 0.5, 1.], [1.0, 0.5, 0., -0.5, -1.]], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)
    '''
    max_nbits = dtype().itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    sign = np.signbit(x)
    enc = encode_fp_to_unorm(np.fabs(x), dtype=dtype, nbits=nbits-1)
    return sign | (enc << dtype(1))

def decode_fp_from_snorm(x, *, dtype=np.float32, nbits=None):
    '''Decode floating-points from signed normalized integers.

    Args:
        x: The type should be `np.uint`, or an array in `np.uint`.
        dtype: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.

    Examples:
        >>> fp = np.float32(-1.)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_fp_from_snorm(enc, dtype=np.float32, nbits=16)

        >>> fp = np.array([-1.0, -0.5, 0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_fp_from_snorm(enc, dtype=np.float32, nbits=16)

        >>> fp = np.array([[-1.0, -0.5, 0., 0.5, 1.], [1.0, 0.5, 0., -0.5, -1.]], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_fp_from_snorm(enc, dtype=np.float32, nbits=16)
    '''
    max_nbits = x.itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    dec = decode_fp_from_unorm(x >> x.dtype.type(1), dtype=dtype, nbits=nbits-1)
    sign = dtype(x & 0x1) * dtype(-2.) + dtype(1.)
    return dec * sign
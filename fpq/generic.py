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
    assert (x.dtype.kind == 'f'), '`dtype` of the argument `x` must be floating point types.'
    assert (dtype().dtype.kind == 'u'), '`dtype` of the argument `dtype` must be unsigned integer types.'

    max_nbits = dtype().itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    assert (nbits <= (np.finfo(x.dtype).bits - np.finfo(x.dtype).iexp)), \
        'Can\'t be expressed with the specified number of bits.'

    max_uint = dtype(np.iinfo(dtype).max) >> dtype(max_nbits - nbits)
    return dtype(np.around(x * x.dtype.type(max_uint)))

def decode_unorm_to_fp(x, *, dtype=np.float32, nbits=None):
    '''Decode unsigned normalized integers to floating-points.

    Args:
        x: The type should be `np.uint`, or an array in `np.uint`.
        dtype: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.

    Examples:
        >>> fp = np.float32(1.)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_unorm_to_fp(enc, dtype=np.float32, nbits=16)

        >>> fp = np.array([0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_unorm_to_fp(enc, dtype=np.float32, nbits=16)

        >>> fp = np.array([[0., 0.5, 1.], [1., 0.5, 0.]], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_unorm_to_fp(enc, dtype=np.float32, nbits=16)
    '''
    assert (x.dtype.kind == 'u'), '`dtype` of the argument `x` must be unsigned integer types.'
    assert (dtype().dtype.kind == 'f'), '`dtype` of the argument `dtype` must be floating point types.'

    max_nbits = x.itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (0 < nbits <= max_nbits), '`nbits` value is out of range.'
    assert (nbits <= (np.finfo(dtype).bits - np.finfo(dtype).iexp)), \
        'Can\'t be expressed with the specified number of bits.'

    max_uint = x.dtype.type(np.iinfo(x.dtype).max) >> x.dtype.type(max_nbits - nbits)
    return dtype(x) / dtype(max_uint)

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
    assert (x.dtype.kind == 'f'), '`dtype` of the argument `x` must be floating point types.'
    assert (dtype().dtype.kind == 'u'), '`dtype` of the argument `dtype` must be unsigned integer types.'

    max_nbits = dtype().itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (1 < nbits <= max_nbits), '`nbits` value is out of range.'
    assert (nbits <= (np.finfo(x.dtype).bits - np.finfo(x.dtype).iexp + 1)), \
        'Can\'t be expressed with the specified number of bits.'

    sign = np.signbit(x)
    enc = encode_fp_to_unorm(np.fabs(x), dtype=dtype, nbits=nbits-1)
    return sign | (enc << dtype(1))

def decode_snorm_to_fp(x, *, dtype=np.float32, nbits=None):
    '''Decode signed normalized integers to floating-points.

    Args:
        x: The type should be `np.uint`, or an array in `np.uint`.
        dtype: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.

    Examples:
        >>> fp = np.float32(-1.)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_snorm_to_fp(enc, dtype=np.float32, nbits=16)

        >>> fp = np.array([-1.0, -0.5, 0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_snorm_to_fp(enc, dtype=np.float32, nbits=16)

        >>> fp = np.array([[-1.0, -0.5, 0., 0.5, 1.], [1.0, 0.5, 0., -0.5, -1.]], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, dtype=np.uint16, nbits=16)
        >>> dec = decode_snorm_to_fp(enc, dtype=np.float32, nbits=16)
    '''
    assert (x.dtype.kind == 'u'), '`dtype` of the argument `x` must be unsigned integer types.'
    assert (dtype().dtype.kind == 'f'), '`dtype` of the argument `dtype` must be floating point types.'

    max_nbits = x.itemsize * 8
    if nbits is None:
        nbits = max_nbits
    assert (1 < nbits <= max_nbits), '`nbits` value is out of range.'
    assert (nbits <= (np.finfo(dtype).bits - np.finfo(dtype).iexp + 1)), \
        'Can\'t be expressed with the specified number of bits.'

    dec = decode_unorm_to_fp(x >> x.dtype.type(1), dtype=dtype, nbits=nbits-1)
    sign = dtype(x & x.dtype.type(0x1)) * dtype(-2.) + dtype(1.)
    return dec * sign

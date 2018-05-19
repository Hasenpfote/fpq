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

    Examples:
        >>> fp = np.float32(1.)
        >>> enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)

        >>> fp = np.array([0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)

        >>> fp = np.array([[0., 0.5, 1.], [1., 0.5, 0.]], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
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

    Examples:
        >>> fp = np.float32(1.)
        >>> enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
        >>> dec = decode_fp_from_unorm(enc, np.float32(), nbits=16)

        >>> fp = np.array([0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
        >>> dec = decode_fp_from_unorm(enc, np.float32(), nbits=16)

        >>> fp = np.array([[0., 0.5, 1.], [1., 0.5, 0.]], dtype=np.float32)
        >>> enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
        >>> dec = decode_fp_from_unorm(enc, np.float32(), nbits=16)
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

    Examples:
        >>> fp = np.float32(-1.)
        >>> enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16)

        >>> fp = np.array([-1.0, -0.5, 0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16)

        >>> fp = np.array([[-1.0, -0.5, 0., 0.5, 1.], [1.0, 0.5, 0., -0.5, -1.]], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16)
    '''
    assert (0 < nbits <= (dst.itemsize * 8)), '`nbits` value is out of range.'
    sign = np.signbit(src)
    enc = encode_fp_to_unorm(np.fabs(src), dst, nbits=nbits-1)
    return sign | (enc << dst.dtype.type(1))

def decode_fp_from_snorm(src, dst, *, nbits):
    '''Decode floating-points from signed normalized integers.

    Args:
        src: The type should be `np.uint`, or an array in `np.uint`.
        dst: The type should be `np.float`.
        nbits: The number of `src` bits to use.

    Returns:
        The resulting floating-points.

    Examples:
        >>> fp = np.float32(-1.)
        >>> enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16)
        >>> dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)

        >>> fp = np.array([-1.0, -0.5, 0., 0.5, 1.], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16)
        >>> dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)

        >>> fp = np.array([[-1.0, -0.5, 0., 0.5, 1.], [1.0, 0.5, 0., -0.5, -1.]], dtype=np.float32)
        >>> enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16)
        >>> dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)
    '''
    assert (0 < nbits <= (src.itemsize * 8)), '`nbits` value is out of range.'
    dec = decode_fp_from_unorm(src >> src.dtype.type(1), dst, nbits=nbits-1)
    sign = dst.dtype.type(src & 0x1) * dst.dtype.type(-2.) + dst.dtype.type(1.)
    return dec * sign

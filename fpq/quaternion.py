#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from . import utils
from . import fpq

def get_max_component_indices(x):
    '''Get the maximum component indices.'''
    ind1 = x.argmax(axis=x.ndim-1)
    ind2 = np.indices(ind1.shape)
    shape = (ind2.shape[0]+1,) + ind2.shape[1:]
    return tuple(np.append(ind2, ind1).reshape(shape))

def remove_max_component(x):
    '''Remove the maximum component.'''
    ma = np.ma.array(x, mask=False)
    ind = get_max_component_indices(x)
    ma.mask[ind] = True
    shape = x.shape[:-1] + (x.shape[-1] - 1,)
    return ma.compressed().reshape(shape)

def encode_quat_to_uint32(q, *, encoder=fpq.encode_fp_to_snorm):
    '''Encode Quaternion to uint32.

    Args:
        q: Should be represented by four components of float32, or an array of them.
        encoder: This is a function encodes a floating point to an unsigned integer.

    Returns:
        The resulting uint32.
    '''
    src_dtype = np.float32
    dst_dtype = np.uint32
    assert (isinstance(q, src_dtype)
            or (isinstance(q, np.ndarray) and (q.dtype == src_dtype))), \
        'The type should be {0}, or an array in {0}.'.format(src_dtype().dtype.name)

    abs_q = np.fabs(q)
    max_abs_ind = get_max_component_indices(abs_q)
    sign = np.where(q[max_abs_ind] < 0., -1., 1.)
    remaining = sign[..., None] * remove_max_component(abs_q)

    # [-1/sqrt(2), +1/sqrt(2)] -> [-1, +1]
    src_max = np.reciprocal(np.sqrt(src_dtype(2.)))
    src_min = -src_max
    remapped = utils.remap(remaining, src_min, src_max, src_dtype(-1.), src_dtype(1.))

    enc = encoder(remapped, dtype=dst_dtype, nbits=10)

    return (dst_dtype(max_abs_ind[-1]) << dst_dtype(30)) | (enc[..., 0] << dst_dtype(20)) \
           | (enc[..., 1] << dst_dtype(10)) | enc[..., 2]

def decode_quat_from_uint32(q, *, decoder=fpq.decode_fp_from_snorm):
    '''Decode Quaternion from uint32.

    Args:
        q: Should be represented by uint32, or an array of them.
        decoder: This is a function decodes an unsigned integer to a floating point.

    Returns:
        The resulting Quaternion.
    '''
    src_dtype = np.uint32
    dst_dtype = np.float32
    assert (isinstance(q, src_dtype)
            or (isinstance(q, np.ndarray) and (q.dtype == src_dtype))), \
        'The type should be {0}, or an array in {0}.'.format(src_dtype().dtype.name)

    mask = src_dtype(0x3ff)
    c1 = decoder((q >> src_dtype(20)) & mask, dtype=dst_dtype, nbits=10)
    c2 = decoder((q >> src_dtype(10)) & mask, dtype=dst_dtype, nbits=10)
    c3 = decoder(q & mask, dtype=dst_dtype, nbits=10)

    # [-1/sqrt(2), +1/sqrt(2)] -> [-1, +1]
    src_max = np.reciprocal(np.sqrt(dst_dtype(2.)))
    src_min = -src_max
    c1 = utils.remap(c1, dst_dtype(-1.), dst_dtype(1.), src_min, src_max)
    c2 = utils.remap(c2, dst_dtype(-1.), dst_dtype(1.), src_min, src_max)
    c3 = utils.remap(c3, dst_dtype(-1.), dst_dtype(1.), src_min, src_max)
    c0 = np.sqrt(dst_dtype(1.) - np.square(c1) - np.square(c2) - np.square(c3))

    max_c = q >> src_dtype(30)
    return np.where(max_c == 0, (c0, c1, c2, c3),
                    np.where(max_c == 1, (c1, c0, c2, c3),
                             np.where(max_c == 2, (c1, c2, c0, c3), (c1, c2, c3, c0)))).transpose()

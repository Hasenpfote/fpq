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

def remove_component(x, *, indices):
    '''Removes a component at the specified index.'''
    ma = np.ma.array(x, mask=False)
    ma.mask[indices] = True
    shape = x.shape[:-1] + (x.shape[-1] - 1,)
    return ma.compressed().reshape(shape)

def encode_quat_to_uint(q, *, dtype=np.uint64, encoder=fpq.encode_fp_to_snorm):
    '''Encode Quaternions to unsigned integers.

    Args:
        q: Should be represented by four components of float, or an array of them.
        dtype: The type should be unsigned integer types.
        encoder: This is a function encodes a floating point to an unsigned integer.

    Returns:
        The resulting unsigned integers.
    '''
    assert (isinstance(q, np.ndarray) and (q.dtype.kind == 'f')), \
        '`dtype` of the argument `q` should be floating point types.'
    assert (dtype().dtype.kind == 'u'), \
        '`dtype` of the argument `dtype` should be unsigned integer types.'

    nbits_per_component = ((dtype().dtype.itemsize * 8) - 2) // 3

    max_abs_ind = get_max_component_indices(np.fabs(q))
    sign = np.where(q[max_abs_ind] < 0., -1., 1.)
    remaining = sign[..., None] * remove_component(q, indices=max_abs_ind)

    # [-1/sqrt(2), +1/sqrt(2)] -> [-1, +1]
    src_max = np.reciprocal(np.sqrt(q.dtype.type(2.)))
    src_min = -src_max
    remapped = utils.remap(remaining, src_min, src_max, q.dtype.type(-1.), q.dtype.type(1.))

    enc = encoder(remapped, dtype=dtype, nbits=nbits_per_component)

    return (dtype(max_abs_ind[-1]) << dtype(nbits_per_component * 3)) \
           | (enc[..., 0] << dtype(nbits_per_component * 2)) \
           | (enc[..., 1] << dtype(nbits_per_component)) \
           | enc[..., 2]


def decode_quat_from_uint(q, *, dtype=np.float64, decoder=fpq.decode_fp_from_snorm):
    '''Decode Quaternions from unsigned integers.

    Args:
        q: Should be represented by uint, or an array of them.
        dtype: The type should be floating point types.
        decoder: This is a function decodes an unsigned integer to a floating point.

    Returns:
        The resulting Quaternions.
    '''
    assert (q.dtype.kind == 'u'), \
        '`dtype` of the argument `q` should be unsigned integer types.'
    assert (dtype().dtype.kind == 'f'), \
        '`dtype` of the argument `dtype` should be floating point types.'

    bits_per_component = ((q.dtype.itemsize * 8) - 2) // 3
    mask = np.invert(q.dtype.type(np.iinfo(q.dtype).max) << q.dtype.type(bits_per_component))

    c1 = decoder((q >> q.dtype.type(bits_per_component * 2)) & mask,
                 dtype=dtype, nbits=bits_per_component)
    c2 = decoder((q >> q.dtype.type(bits_per_component)) & mask,
                 dtype=dtype, nbits=bits_per_component)
    c3 = decoder(q & mask, dtype=dtype, nbits=bits_per_component)

    # [-1/sqrt(2), +1/sqrt(2)] -> [-1, +1]
    src_max = np.reciprocal(np.sqrt(dtype(2.)))
    src_min = -src_max
    c1 = utils.remap(c1, dtype(-1.), dtype(1.), src_min, src_max)
    c2 = utils.remap(c2, dtype(-1.), dtype(1.), src_min, src_max)
    c3 = utils.remap(c3, dtype(-1.), dtype(1.), src_min, src_max)
    c0 = np.sqrt(dtype(1.) - np.square(c1) - np.square(c2) - np.square(c3))

    max_c = q >> q.dtype.type(bits_per_component * 3)
    return np.where(max_c == 0, (c0, c1, c2, c3),
                    np.where(max_c == 1, (c1, c0, c2, c3),
                             np.where(max_c == 2, (c1, c2, c0, c3), (c1, c2, c3, c0)))).transpose()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from . import utils
from . import generic


def is_valid_format(dtype_f, dtype_u, nbits):
    assert (dtype_f().dtype.kind == 'f'), \
        '`dtype` of the argument `dtype_f` must be floating point types.'
    assert (dtype_u().dtype.kind == 'u'), \
        '`dtype` of the argument `dtype_ui` must be unsigned integer types.'

    remaining = dtype_u().itemsize * 8 - 2
    if (nbits < 2) or (nbits > ((remaining - 2) // 2)):
        return False
    if (remaining - nbits * 2) > (dtype_f().itemsize * 8):
        return False
    return nbits <= (2 + np.finfo(dtype_f).nmant)


def calc_breakdown_of_uint(dtype, nbits):
    '''Calculate a breakdown of an unsigned integer.'''
    assert (dtype().dtype.kind == 'u'), \
        '`dtype` of the argument `dtype` must be unsigned integer types.'

    remaining = dtype().itemsize * 8 - 2
    return 2, nbits, nbits, remaining - nbits * 2


def encode_vec_to_uint(v, *, dtype=np.uint64, nbits=20, encoder=generic.encode_fp_to_snorm):
    assert is_valid_format(v.dtype.type, dtype, nbits), 'Not a valid format.'

    # Get the maximum absolute component indices.
    max_abs_ind = utils.get_max_component_indices(np.fabs(v))

    # Normalize the vectors.
    norm = np.linalg.norm(v, axis=-1)
    nv = v / norm[..., None]

    # The sign of the maximum absolute component.
    sign = np.where(nv[max_abs_ind] < 0., -1., 1.)

    # Removes the maximum absolute component, and apply the sign.
    remaining = utils.remove_component(nv, indices=max_abs_ind) * sign[..., None]

    #
    breakdown = calc_breakdown_of_uint(dtype, nbits)

    # Encoding for vector components.
    enc = encoder(remaining, dtype=dtype, nbits=breakdown[1])

    # Encoding for the vector norm.
    norm *= sign
    if (breakdown[3] <= 16) and (norm.dtype != np.float16):
        norm = np.float16(norm)
    elif (breakdown[3] <= 32) and (norm.dtype != np.float32):
        norm = np.float32(norm)
    elif norm.dtype != np.float64:
        norm = np.float64(norm)
    enc_n = generic.encode_fp_to_uint(norm, nbits=breakdown[3])
    if enc_n.dtype != dtype:
        enc_n = dtype(enc_n)

    return (dtype(max_abs_ind[-1]) << dtype(sum(breakdown[1:]))) \
           | (enc[..., 0] << dtype(sum(breakdown[2:]))) \
           | (enc[..., 1] << dtype(sum(breakdown[3:]))) \
           | enc_n


def decode_uint_to_vec(v, *, dtype=np.float64, nbits=20, decoder=generic.decode_snorm_to_fp):
    assert is_valid_format(dtype, v.dtype.type, nbits), 'Not a valid format.'

    breakdown = calc_breakdown_of_uint(v.dtype.type, nbits)

    shifts = np.array([sum(breakdown[1:]), sum(breakdown[2:]), sum(breakdown[3:])], dtype=v.dtype)
    masks = np.invert(v.dtype.type(np.iinfo(v.dtype).max) << np.array(breakdown[1:], dtype=v.dtype))

    # Decoding for vector components.
    dec_1 = decoder((v >> shifts[1]) & masks[0], dtype=dtype, nbits=breakdown[1])
    dec_2 = decoder((v >> shifts[2]) & masks[1], dtype=dtype, nbits=breakdown[2])

    # Decoding for the vector norm.
    norm = v & masks[2]
    if (breakdown[3] <= 16) and (norm.dtype != np.uint16):
        norm = np.uint16(norm)
    elif (breakdown[3] <= 32) and (norm.dtype != np.uint32):
        norm = np.uint32(norm)
    elif norm.dtype != np.uint64:
        norm = np.uint64(norm)
    dec_n = generic.decode_uint_to_fp(norm, nbits=breakdown[3])
    if dec_n.dtype != dtype:
        dec_n = dtype(dec_n)

    dec_0 = np.sqrt(dtype(1.) - np.square(dec_1) - np.square(dec_2))

    c0 = dec_0 * dec_n
    c1 = dec_1 * dec_n
    c2 = dec_2 * dec_n

    max_c = v >> v.dtype.type(shifts[0])
    return np.where(max_c == 0, (c0, c1, c2),
                    np.where(max_c == 1, (c1, c0, c2), (c1, c2, c0))).transpose()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase
import numpy as np
import sys
sys.path.append('../')
from fpq.d3dfpq import *


class TestD3DFpq(TestCase):
    def test_encode_fp_to_unorm(self):
        nbits = 5
        dst = np.uint8()

        fp = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.], dtype=np.float32)
        enc = encode_fp_to_unorm(fp, dst, nbits=nbits)
        self.assertTrue(isinstance(enc, np.ndarray) and (enc.dtype == dst.dtype) and (enc.shape == fp.shape))
        # c * (2^n - 1) + 0.5
        self.assertTrue(np.array_equal(enc, [0, 6, 12, 19, 25, 31]))

    def test_decode_fp_from_unorm(self):
        nbits = 5
        dst = np.float32()

        enc = np.array([0, 6, 12, 19, 25, 31], dtype=np.uint8)
        dec = decode_fp_from_unorm(enc, dst, nbits=nbits)
        self.assertTrue(isinstance(dec, np.ndarray) and (dec.dtype == dst.dtype) and (dec.shape == enc.shape))
        self.assertTrue(np.allclose(dec, [0., 0.2, 0.4, 0.6, 0.8, 1.], rtol=0., atol=1e-1))

    def test_encode_fp_to_snorm(self):
        nbits = 5
        dst = np.uint8()

        fp = np.array([-1., -0.9, -0., 0., 0.9, 1.], dtype=np.float32)
        enc = encode_fp_to_snorm(fp, dst, nbits=nbits)
        self.assertTrue(isinstance(enc, np.ndarray) and (enc.dtype == dst.dtype) and (enc.shape == fp.shape))
        # c * (2^(n-1) - 1)
        # c >= 0: c + 0.5
        # c <  0: c - 0.5
        self.assertTrue(np.array_equal(enc, [16, 17, 31, 0, 14, 15]))

    def test_decode_fp_from_snorm(self):
        nbits = 5
        dst = np.float32()

        enc = np.array([16, 17, 31, 0, 14, 15], dtype=np.uint8)
        dec = decode_fp_from_snorm(enc, dst, nbits=nbits)
        self.assertTrue(isinstance(dec, np.ndarray) and (dec.dtype == dst.dtype) and (dec.shape == enc.shape))
        self.assertTrue(np.allclose(dec, [-1., -1., -0., 0., 0.9, 1.], rtol=0., atol=1e-1))

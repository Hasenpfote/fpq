#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase
import numpy as np
import sys
sys.path.append('../')
from fpq.fpq import *


class TestFpq(TestCase):
    def test_encode_fp_to_unorm_by_0dim(self):
        fp = np.float32(0.)
        enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
        self.assertTrue(isinstance(enc, np.uint16))
        self.assertEqual(enc, 0)

        fp = np.float32(0.5)
        enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
        self.assertTrue(isinstance(enc, np.uint16))
        self.assertEqual(enc, 32768)

        fp = np.float32(1.)
        enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
        self.assertTrue(isinstance(enc, np.uint16))
        self.assertEqual(enc, 65535)

    def test_decode_fp_from_unorm_by_0dim(self):
        enc = np.uint16(0)
        dec = decode_fp_from_unorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.float32))
        self.assertTrue(np.isclose(dec, 0., rtol=0., atol=1e-5))

        enc = np.uint16(32768)
        dec = decode_fp_from_unorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.float32))
        self.assertTrue(np.isclose(dec, 0.5, rtol=0., atol=1e-5))

        enc = np.uint16(65535)
        dec = decode_fp_from_unorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.float32))
        self.assertTrue(np.isclose(dec, 1.0, rtol=0., atol=1e-5))

    def test_encode_fp_to_snorm_by_0dim(self):
        fp = np.float32(-1.)
        enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16) # 65535
        self.assertTrue(isinstance(enc, np.uint16))
        self.assertEqual(enc, 65535)

        fp = np.float32(-0.5)
        enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16) # 32769
        self.assertTrue(isinstance(enc, np.uint16))
        self.assertEqual(enc, 32769)

        fp = np.float32(0.)
        enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16) # 0
        self.assertTrue(isinstance(enc, np.uint16))
        self.assertEqual(enc, 0)

        fp = np.float32(0.5)
        enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16) # 32768
        self.assertTrue(isinstance(enc, np.uint16))
        self.assertEqual(enc, 32768)

        fp = np.float32(1.)
        enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16) # 65534
        self.assertTrue(isinstance(enc, np.uint16))
        self.assertEqual(enc, 65534)

    def test_decode_fp_from_snorm_by_0dim(self):
        enc = np.uint16(65535)
        dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.float32))
        self.assertTrue(np.isclose(dec, -1., rtol=0., atol=1e-4))

        enc = np.uint16(32769)
        dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.float32))
        self.assertTrue(np.isclose(dec, -0.5, rtol=0., atol=1e-4))

        enc = np.uint16(0)
        dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.float32))
        self.assertTrue(np.isclose(dec, 0., rtol=0., atol=1e-4))

        enc = np.uint16(32768)
        dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.float32))
        self.assertTrue(np.isclose(dec, 0.5, rtol=0., atol=1e-4))

        enc = np.uint16(65534)
        dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.float32))
        self.assertTrue(np.isclose(dec, 1.0, rtol=0., atol=1e-4))

    def test_encode_fp_to_unorm_by_1dim(self):
        fp = np.array([0., 0.5, 1.], dtype=np.float32)
        enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
        self.assertTrue(isinstance(enc, np.ndarray) and (enc.dtype == np.uint16) and (enc.shape == fp.shape))
        self.assertTrue(np.array_equal(enc, [0, 32768, 65535]))

    def test_decode_fp_from_unorm_by_1dim(self):
        enc = np.array([0, 32768, 65535], dtype=np.uint16)
        dec = decode_fp_from_unorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.ndarray) and (dec.dtype == np.float32) and (dec.shape == enc.shape))
        self.assertTrue(np.allclose(dec, [0., 0.5, 1.], rtol=0., atol=1e-5))

    def test_encode_fp_to_snorm_by_1dim(self):
        fp = np.array([-1., -0.5, 0, 0.5, 1.], dtype=np.float32)
        enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16)
        self.assertTrue(isinstance(enc, np.ndarray) and (enc.dtype == np.uint16) and (enc.shape == fp.shape))
        self.assertTrue(np.array_equal(enc, [65535, 32769, 0, 32768, 65534]))

    def test_decode_fp_from_snorm_by_1dim(self):
        enc = np.array([65535, 32769, 0, 32768, 65534], dtype=np.uint16)
        dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.ndarray) and (dec.dtype == np.float32) and (dec.shape == enc.shape))
        self.assertTrue(np.allclose(dec, [-1., -0.5, 0, 0.5, 1.], rtol=0., atol=1e-4))

    def test_encode_fp_to_unorm_by_ndim(self):
        fp = np.array([[0., 0.5, 1.],
                       [1., 0.5, 0.]], dtype=np.float32)
        enc = encode_fp_to_unorm(fp, np.uint16(), nbits=16)
        self.assertTrue(isinstance(enc, np.ndarray) and (enc.dtype == np.uint16) and (enc.shape == fp.shape))
        self.assertTrue(np.array_equal(enc, [[0, 32768, 65535],
                                             [65535, 32768, 0]]))

    def test_decode_fp_from_unorm_by_ndim(self):
        enc = np.array([[0, 32768, 65535],
                        [65535, 32768, 0]], dtype=np.uint16)
        dec = decode_fp_from_unorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.ndarray) and (dec.dtype == np.float32) and (dec.shape == enc.shape))
        self.assertTrue(np.allclose(dec, [[0., 0.5, 1.],
                                          [1., 0.5, 0.]], rtol=0., atol=1e-5))

    def test_encode_fp_to_snorm_by_ndim(self):
        fp = np.array([[-1., -0.5, 0, 0.5, 1.],
                       [1., 0.5, 0, -0.5, -1.]], dtype=np.float32)
        enc = encode_fp_to_snorm(fp, np.uint16(), nbits=16)
        self.assertTrue(isinstance(enc, np.ndarray) and (enc.dtype == np.uint16) and (enc.shape == fp.shape))
        self.assertTrue(np.array_equal(enc, [[65535, 32769, 0, 32768, 65534],
                                             [65534, 32768, 0, 32769, 65535]]))

    def test_decode_fp_from_snorm_by_ndim(self):
        enc = np.array([[65535, 32769, 0, 32768, 65534],
                        [65534, 32768, 0, 32769, 65535]], dtype=np.uint16)
        dec = decode_fp_from_snorm(enc, np.float32(), nbits=16)
        self.assertTrue(isinstance(dec, np.ndarray) and (dec.dtype == np.float32) and (dec.shape == enc.shape))
        self.assertTrue(np.allclose(dec, [[-1., -0.5, 0, 0.5, 1.],
                                          [1., 0.5, 0, -0.5, -1.]], rtol=0., atol=1e-4))

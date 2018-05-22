#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase
import numpy as np
import sys
sys.path.append('../')
from fpq.generic import *


class TestGeneric(TestCase):
    def test_encode_fp_to_unorm(self):
        fp = np.array([0., 0.25, 1.], dtype=np.float32)
        actual = encode_fp_to_unorm(fp, dtype=np.uint8, nbits=5)
        expected = np.array([0b00000, 0b01000, 0b11111], dtype=np.uint8)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.array_equal(actual, expected))

        fp = np.array([0., 1./3., 2./3., 1.], dtype=np.float32)
        actual = encode_fp_to_unorm(fp, dtype=np.uint8, nbits=2)
        expected = np.array([0, 1, 2, 3], dtype=np.uint8)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.array_equal(actual, expected))

    def test_decode_unorm_to_fp(self):
        enc = np.array([0b00000, 0b01000, 0b11111], dtype=np.uint8)
        actual = decode_unorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.array([0., 0.25, 1.], dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.allclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.array([0, 1, 2, 3], dtype=np.uint8)
        actual = decode_unorm_to_fp(enc, dtype=np.float32, nbits=2)
        expected = np.array([0., 1./3., 2./3., 1.], dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.allclose(actual, expected, rtol=0., atol=1e-1))

    def test_encode_fp_to_snorm(self):
        fp = np.array([-1., -0.25, -0., 0., 0.25, 1.], dtype=np.float32)
        actual = encode_fp_to_snorm(fp, dtype=np.uint8, nbits=5)
        expected = np.array([0b11111, 0b01001, 0b00001, 0b00000, 0b01000, 0b11110], dtype=np.uint8)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.array_equal(actual, expected))

    def test_decode_snorm_to_fp(self):
        enc = np.array([0b11111, 0b01001, 0b00001, 0b00000, 0b01000, 0b11110], dtype=np.uint8)
        actual = decode_snorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.array([-1., -0.25, -0., 0., 0.25, 1.], dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.allclose(actual, expected, rtol=0., atol=1e-1))

    def test_encode_fp_to_unorm_by_scalar(self):
        fp = np.float32(0.)
        actual = encode_fp_to_unorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b00000)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

        fp = np.float32(0.25)
        actual = encode_fp_to_unorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b01000)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

        fp = np.float32(1.)
        actual = encode_fp_to_unorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b11111)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

    def test_decode_unorm_to_fp_by_scalar(self):
        enc = np.uint8(0b00000)
        actual = decode_unorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(0.)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.uint8(0b01000)
        actual = decode_unorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(0.25)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.uint8(0b11111)
        actual = decode_unorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(1.)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))

    def test_encode_fp_to_snorm_by_scalar(self):
        fp = np.float32(-1.)
        actual = encode_fp_to_snorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b11111)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

        fp = np.float32(-0.25)
        actual = encode_fp_to_snorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b01001)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

        fp = np.float32(-0.)
        actual = encode_fp_to_snorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b00001)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

        fp = np.float32(0.)
        actual = encode_fp_to_snorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b00000)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

        fp = np.float32(0.25)
        actual = encode_fp_to_snorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b01000)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

        fp = np.float32(1.)
        actual = encode_fp_to_snorm(fp, dtype=np.uint8, nbits=5)
        expected = np.uint8(0b11110)
        self.assertTrue(isinstance(actual, np.uint8))
        self.assertTrue(np.array_equal(actual, expected))

    def test_decode_snorm_to_fp_by_scalar(self):
        enc = np.uint8(0b11111)
        actual = decode_snorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(-1.)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.uint8(0b01001)
        actual = decode_snorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(-0.25)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.uint8(0b00001)
        actual = decode_snorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(-0.)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.uint8(0b00000)
        actual = decode_snorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(0.)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.uint8(0b01000)
        actual = decode_snorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(0.25)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.uint8(0b11110)
        actual = decode_snorm_to_fp(enc, dtype=np.float32, nbits=5)
        expected = np.float32(1.)
        self.assertTrue(isinstance(actual, np.float32))
        self.assertTrue(np.isclose(actual, expected, rtol=0., atol=1e-1))


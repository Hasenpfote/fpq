#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase
import numpy as np
import sys
sys.path.append('../')
from fpq.d3dfpq import *


class TestD3DFpq(TestCase):
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

    def test_decode_fp_from_unorm(self):
        enc = np.array([0b00000, 0b01000, 0b11111], dtype=np.uint8)
        actual = decode_fp_from_unorm(enc, dtype=np.float32, nbits=5)
        expected = np.array([0., 0.25, 1.], dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.allclose(actual, expected, rtol=0., atol=1e-1))

        enc = np.array([0, 1, 2, 3], dtype=np.uint8)
        actual = decode_fp_from_unorm(enc, dtype=np.float32, nbits=2)
        expected = np.array([0., 1./3., 2./3., 1.], dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.allclose(actual, expected, rtol=0., atol=1e-1))

    def test_encode_fp_to_snorm(self):
        fp = np.array([-1., -0.25, -0., 0., 0.25, 1.], dtype=np.float32)
        actual = encode_fp_to_snorm(fp, dtype=np.uint8, nbits=5)
        expected = np.array([0b10001, 0b11100, 0b00000, 0b00000, 0b00100, 0b01111], dtype=np.uint8)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.array_equal(actual, expected))

    def test_decode_fp_from_snorm(self):
        enc = np.array([0b10000, 0b10001, 0b11100, 0b00000, 0b00100, 0b01111], dtype=np.uint8)
        actual = decode_fp_from_snorm(enc, dtype=np.float32, nbits=5)
        expected = np.array([-1., -1., -0.25, 0., 0.25, 1.], dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray)
                        and (actual.dtype == expected.dtype)
                        and (actual.shape == expected.shape))
        self.assertTrue(np.allclose(actual, expected, rtol=0., atol=1e-1))

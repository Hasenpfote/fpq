#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase
import math
import numpy as np
import sys
sys.path.append('../')
from fpq.quaternion import *
import fpq.d3dfpq


class TestQuaternion(TestCase):
    @staticmethod
    def quat_from_axis_angle(axis, angle):
        axis_ = np.array(axis, dtype=np.float64)
        half_angle = angle * 0.5
        ret = np.empty(4)
        ret[0] = math.cos(half_angle)
        ret[1:4] = math.sin(half_angle) * axis_
        return ret

    @staticmethod
    def quat_are_same_rotation(q1, q2, *, atol=1e-08):
        return np.isclose(1., abs(np.dot(q1, q2)), rtol=0., atol=atol)

    def test_get_max_component_indices(self):
        arr = np.array([1., 2., 3., 4.])
        actual = get_max_component_indices(arr)
        expected = (3,)
        self.assertTrue(isinstance(actual, tuple))
        self.assertTrue(np.array_equal(actual, expected))

        arr = np.array([[1., 2., 3., 4.],
                        [3., 2., 1., 0.]])
        actual = get_max_component_indices(arr)
        expected = (np.array([0, 1]), np.array([3, 0]))
        self.assertTrue(isinstance(actual, tuple))
        self.assertTrue(np.array_equal(actual, expected))

    def test_remove_max_component(self):
        arr = np.array([1., 2., 3., 4.])
        actual = remove_max_component(arr)
        expected = np.array([1., 2., 3.])
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(np.array_equal(actual, expected))

        arr = np.array([[1., 2., 3., 4.],
                        [3., 2., 1., 0.]])
        actual = remove_max_component(arr)
        expected = np.array([[1., 2., 3.],
                             [2., 1., 0.]])
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(np.array_equal(actual, expected))

    def test_encode_quat_to_uint32(self):
        q = np.array(self.quat_from_axis_angle([1., 0., 0.], math.radians(350.)), dtype=np.float32)
        actual = encode_quat_to_uint(q, dtype=np.uint32)
        expected = np.uint32(133169152)
        self.assertTrue(isinstance(actual, np.uint32))
        self.assertTrue(np.array_equal(actual, expected))

        q = np.array([self.quat_from_axis_angle([1., 0., 0.], math.radians(350.)),   # max: |w|
                      self.quat_from_axis_angle([1., 0., 0.], math.radians(-120.)),  # max: |x|
                      self.quat_from_axis_angle([0., 1., 0.], math.radians(-120.)),  # max: |y|
                      self.quat_from_axis_angle([0., 0., 1.], math.radians(-120.))], # max: |z|
                     dtype=np.float32)
        actual = encode_quat_to_uint(q, dtype=np.uint32)
        expected = np.array([133169152, 1831862272, 2905604096, 3979345920], dtype=np.uint32)
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(actual.dtype == np.uint32)

    def test_decode_quat_from_uint32(self):
        q = np.uint32(133169152)
        actual = decode_quat_from_uint(q, dtype=np.float32)
        expected = np.array(self.quat_from_axis_angle([1., 0., 0.], math.radians(350.)), dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(actual.dtype == np.float32)
        self.assertTrue(self.quat_are_same_rotation(actual, expected, atol=1e-06))

        q = np.array([133169152, 1831862272, 2905604096, 3979345920], dtype=np.uint32)
        actual = decode_quat_from_uint(q, dtype=np.float32)
        expected = np.array([self.quat_from_axis_angle([1., 0., 0.], math.radians(350.)),   # max: |w|
                             self.quat_from_axis_angle([1., 0., 0.], math.radians(-120.)),  # max: |x|
                             self.quat_from_axis_angle([0., 1., 0.], math.radians(-120.)),  # max: |y|
                             self.quat_from_axis_angle([0., 0., 1.], math.radians(-120.))], # max: |z|
                            dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(actual.dtype == np.float32)
        self.assertTrue(self.quat_are_same_rotation(actual[0], expected[0], atol=1e-06))
        self.assertTrue(self.quat_are_same_rotation(actual[1], expected[1], atol=1e-06))
        self.assertTrue(self.quat_are_same_rotation(actual[2], expected[2], atol=1e-06))
        self.assertTrue(self.quat_are_same_rotation(actual[3], expected[3], atol=1e-06))

    def test_encode_quat_to_uint32_by_d3dfpq(self):
        encoder = fpq.d3dfpq.encode_fp_to_snorm

        q = np.array(self.quat_from_axis_angle([1., 0., 0.], math.radians(350.)), dtype=np.float32)
        actual = encode_quat_to_uint(q, dtype=np.uint32, encoder=encoder)
        expected = np.uint32(1007681536)
        self.assertTrue(isinstance(actual, np.uint32))
        self.assertTrue(np.array_equal(actual, expected))

    def test_decode_quat_from_uint32_by_d3dfpq(self):
        decoder = fpq.d3dfpq.decode_fp_from_snorm

        q = np.uint32(1007681536)
        actual = decode_quat_from_uint(q, dtype=np.float32, decoder=decoder)
        expected = np.array(self.quat_from_axis_angle([1., 0., 0.], math.radians(350.)), dtype=np.float32)
        self.assertTrue(isinstance(actual, np.ndarray))
        self.assertTrue(actual.dtype == np.float32)
        self.assertTrue(self.quat_are_same_rotation(actual, expected, atol=1e-06))

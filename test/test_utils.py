#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest import TestCase
import numpy as np
import sys
sys.path.append('../')
from fpq.utils import *


class TestUtils(TestCase):
    def test_remap(self):
        src_min, src_max = 0., 10.
        dst_min, dst_max = 0., 1.
        src_val = 10.
        dst_val = remap(src_val, src_min, src_max, dst_min, dst_max)
        self.assertTrue(isinstance(dst_val, float))
        self.assertTrue(dst_min <= dst_val <= dst_max)

        src_min, src_max = -10., 10.
        dst_min, dst_max = -1., 1.
        src_val = -10.
        dst_val = remap(src_val, src_min, src_max, dst_min, dst_max)
        self.assertTrue(isinstance(dst_val, float))
        self.assertTrue(dst_min <= dst_val <= dst_max)

        src_min, src_max = 0., 10.
        dst_min, dst_max = 0., 1.
        src_val = np.array([0., 2.5, 5., 7.5, 10.])
        dst_val = remap(src_val, src_min, src_max, dst_min, dst_max)
        self.assertTrue(isinstance(dst_val, np.ndarray))
        self.assertTrue(np.all(dst_val >= dst_min) and np.all(dst_val <= dst_max))

        src_min, src_max = -10., 10.
        dst_min, dst_max = -1., 1.
        src_val = np.array([-10, -7.5, -5., -2.5, 0., 2.5, 5., 7.5, 10.])
        dst_val = remap(src_val, src_min, src_max, dst_min, dst_max)
        self.assertTrue(isinstance(dst_val, np.ndarray))
        self.assertTrue(np.all(dst_val >= dst_min) and np.all(dst_val <= dst_max))

    def test__can_express_norm(self):
        import fpq.utils
        self.assertTrue(fpq.utils._can_express_norm(11, np.float16))
        self.assertFalse(fpq.utils._can_express_norm(12, np.float16))
        self.assertTrue(fpq.utils._can_express_norm(24, np.float32))
        self.assertFalse(fpq.utils._can_express_norm(25, np.float32))
        self.assertTrue(fpq.utils._can_express_norm(53, np.float64))
        self.assertFalse(fpq.utils._can_express_norm(54, np.float64))

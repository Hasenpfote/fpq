#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import random
import numpy as np


def make_unorm_dataset(size, *, dtype):
    return np.random.uniform(low=0., high=1., size=size).astype(dtype)


def make_snorm_dataset(size, *, dtype):
    return np.random.uniform(low=-1., high=1., size=size).astype(dtype)


def make_vec_dataset(size, *, dtype):
    def random_vector(size):
        for _ in range(size):
            phi = random.uniform(0., 2. * math.pi)
            theta = math.acos(random.uniform(-1., 1.))
            yield math.sin(theta) * math.cos(phi)
            yield math.sin(theta) * math.sin(phi)
            yield math.cos(theta)

    dataset = np.fromiter(
        random_vector(size),
        dtype=dtype,
        count=size*3
    )
    return dataset.reshape(size, 3)


def make_quat_dataset(size, *, dtype):
    def random_quat(size):
        for _ in range(size):
            u1 = random.random()
            r1 = math.sqrt(1. - u1)
            r2 = math.sqrt(u1)
            t1 = 2. * math.pi * random.random() # u1
            t2 = 2. * math.pi * random.random() # u2
            yield r2 * math.cos(t2) # w
            yield r1 * math.sin(t1) # x
            yield r1 * math.cos(t1) # y
            yield r2 * math.sin(t2) # z

    dataset = np.fromiter(
        random_quat(size),
        dtype=dtype,
        count=size*4
    )
    return dataset.reshape(size, 4)

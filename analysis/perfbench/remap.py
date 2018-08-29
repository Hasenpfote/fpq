#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from perfbench import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../../')
from fpq import numba_wrapper


def remap(x, src_min, src_max, dst_min, dst_max):
    '''Maps values from [`src_min`, `src_max`]  to [`dst_min`, `dst_max`].
    Args:
        x: The incoming value to be converted.
        src_min: Lower bound of the value current range.
        src_max: Upper bound of the value current range.
        dst_min: Lower bound of the value target range.
        dst_max: Upper bound of the value target range.
    Returns:
        The resulting value.
    '''
    return (x - src_min) * ((dst_max - dst_min) / (src_max - src_min)) + dst_min
    

@numba_wrapper.avoid_mapping_to_py_types
@numba_wrapper.avoid_non_supported_types
@numba_wrapper.jit
def remap_jit(x, src_min, src_max, dst_min, dst_max):
    '''Maps values from [`src_min`, `src_max`]  to [`dst_min`, `dst_max`].
    Args:
        x: The incoming value to be converted.
        src_min: Lower bound of the value current range.
        src_max: Upper bound of the value current range.
        dst_min: Lower bound of the value target range.
        dst_max: Upper bound of the value target range.
    Returns:
        The resulting value.
    '''
    return (x - src_min) * ((dst_max - dst_min) / (src_max - src_min)) + dst_min    


def make_dataset(n, *, dtype):
    return np.random.uniform(low=-1., high=1., size=n).astype(dtype)


def main():
    bm = Benchmark(
        datasets=[
            Dataset(
                factories=[
                    lambda n: make_dataset(n, dtype=np.float16),
                ],
                title='float16'
            ),
            Dataset(
                factories=[
                    lambda n: make_dataset(n, dtype=np.float32),
                ],
                title='float32'
            ),
            Dataset(
                factories=[
                    lambda n: make_dataset(n, dtype=np.float64),
                ],
                title='float64'
            )
        ],
        dataset_sizes=[2 ** n for n in range(25)],
        kernels=[
            Kernel(
                stmt='remap(DATASET, -1., 1., -2., 2.)',
                setup='from __main__ import remap',
                label='remap'
            ),
            Kernel(
                stmt='remap_jit(DATASET, -1., 1., -2., 2.)',
                setup='from __main__ import remap_jit',
                label='remap_jit'
            ),
        ],
        xlabel='dataset sizes',
        title='remap',
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

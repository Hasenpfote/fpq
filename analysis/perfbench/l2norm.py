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


@numba_wrapper.avoid_mapping_to_py_types
@numba_wrapper.avoid_non_supported_types
@numba_wrapper.jit
def l2norm_jit(v):
    '''Calculates the L2 norm.'''
    return np.sqrt(np.square(v[..., 0]) + np.square(v[..., 1]) + np.square(v[..., 2]))


def make_dataset(n, *, dtype):
    return np.random.uniform(low=-100., high=100., size=(n, 3)).astype(dtype)


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
                stmt='np.linalg.norm(DATASET, axis=-1)',
                setup='import numpy as np',
                label='linalg.norm'
            ),
            Kernel(
                stmt='l2norm_jit(DATASET)',
                setup='from __main__ import l2norm_jit',
                label='l2norm_jit'
            ),
            Kernel(
                stmt='np.sqrt(np.square(DATASET[..., 0]) + np.square(DATASET[..., 1]) + np.square(DATASET[..., 2]))',
                setup='import numpy as np',
                label='test1'
            ),
        ],
        xlabel='dataset sizes',
        title='l2norm',
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from perfbench import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../../')
import fpq.numba_wrapper


@fpq.numba_wrapper.avoid_mapping_to_py_types
@fpq.numba_wrapper.avoid_non_supported_types
@fpq.numba_wrapper.jit
def rint_jit(x):
    '''Wrapper function for `numpy.rint`.'''
    return np.rint(x) 

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
                stmt='np.rint(DATASET)',
                setup='import numpy as np',
                label='rint'
            ),
            Kernel(
                stmt='rint_jit(DATASET)',
                setup='from __main__ import rint_jit',
                label='rint_jit'
            ),
        ],
        xlabel='dataset sizes',
        title='rint',
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

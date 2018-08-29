#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from perfbench import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../../')
from data import make_vec_dataset
import fpq.numba_wrapper
import fpq.utils


def _solve_remaining_component(x):
    '''Solve a remaining component of a unit vector'''
    return np.sqrt(x.dtype.type(1.) - np.square(x[0]) - np.square(x[1]))


@fpq.numba_wrapper.avoid_mapping_to_py_types
@fpq.numba_wrapper.avoid_non_supported_types
@fpq.numba_wrapper.jit
def _solve_remaining_component_jit(x):
    '''Solve a remaining component of a unit vector'''
    return np.sqrt(x.dtype.type(1.) - np.square(x[0]) - np.square(x[1]))


def make_dataset(n, *, dtype):
    dataset = make_vec_dataset(n, dtype=dtype)
    max_abs_inds = fpq.utils.get_max_component_indices(np.absolute(dataset))
    sign = np.sign(dataset[max_abs_inds])
    rest_components = fpq.utils.remove_component(dataset, indices=max_abs_inds)
    rest_components *= sign[..., None]
    return rest_components.transpose()


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
                stmt='_solve_remaining_component(DATASET)',
                setup='from __main__ import _solve_remaining_component',
                label='_solve_remaining_component'
            ),
            Kernel(
                stmt='_solve_remaining_component_jit(DATASET)',
                setup='from __main__ import _solve_remaining_component_jit',
                label='_solve_remaining_component_jit'
            ),
            Kernel(
                stmt='np.sqrt(DATASET.dtype.type(1.) - np.sum(np.square(DATASET), axis=0))',
                setup='import numpy as np',
                label='test'
            ),
        ],
        xlabel='dataset sizes',
        title='_solve_remaining_component for Vector',
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

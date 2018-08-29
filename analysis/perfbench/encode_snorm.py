#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from perfbench import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../../')
from data import make_snorm_dataset


def main():
    bm = Benchmark(
        datasets=[
            Dataset(
                factories=[
                    lambda n: make_snorm_dataset(n, dtype=np.float16)
                ],
                title='16bits',
                extra_args=dict(
                    dtype=np.uint16,
                    nbits=4
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_snorm_dataset(n, dtype=np.float32)
                ],
                title='32bits',
                extra_args=dict(
                    dtype=np.uint32,
                    nbits=10
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_snorm_dataset(n, dtype=np.float64)
                ],
                title='64bits',
                extra_args=dict(
                    dtype=np.uint64,
                    nbits=20
                )
            )
        ],
        dataset_sizes=[2 ** n for n in range(25)],
        kernels=[
            Kernel(
                stmt="fp.encode_fp_to_std_snorm(DATASET, dtype=EXTRA_ARGS['dtype'], nbits=EXTRA_ARGS['nbits'])",
                setup='from fpq import fp',
                label='encode_fp_to_std_snorm'
            ),
            Kernel(
                stmt="fp.encode_fp_to_ogl_snorm(DATASET, dtype=EXTRA_ARGS['dtype'], nbits=EXTRA_ARGS['nbits'])",
                setup='from fpq import fp',
                label='encode_fp_to_ogl_snorm'
            ),
        ],
        xlabel='dataset sizes',
        title='encode_fp_to_xxx_snorm'
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

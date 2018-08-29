#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from perfbench import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../../')
from data import make_unorm_dataset
from fpq import fp


def make_dataset(n, *, src_dtype, dst_dtype, nbits):
    return fp.encode_fp_to_std_unorm(make_unorm_dataset(n, dtype=src_dtype), dtype=dst_dtype, nbits=nbits)


def main():
    bm = Benchmark(
        datasets=[
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float16, dst_dtype=np.uint16, nbits=4),
                ],
                title='16bits',
                extra_args=dict(
                    dtype=np.float16,
                    nbits=4
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float32, dst_dtype=np.uint32, nbits=10),
                ],
                title='32bits',
                extra_args=dict(
                    dtype=np.float32,
                    nbits=10
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float64, dst_dtype=np.uint64, nbits=20),
                ],
                title='64bits',
                extra_args=dict(
                    dtype=np.float64,
                    nbits=20
                )
            )
        ],
        dataset_sizes=[2 ** n for n in range(25)],
        kernels=[
            Kernel(
                stmt="fp.decode_std_unorm_to_fp(DATASET, dtype=EXTRA_ARGS['dtype'], nbits=EXTRA_ARGS['nbits'])",
                setup='from fpq import fp',
                label='decode_std_unorm_to_fp'
            ),
        ],
        xlabel='dataset sizes',
        title='decode_std_unorm_to_fp'
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

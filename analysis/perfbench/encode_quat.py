#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from perfbench import *
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../../')
from data import make_quat_dataset


def main():
    bm = Benchmark(
        datasets=[
            Dataset(
                factories=[
                    lambda n: make_quat_dataset(n, dtype=np.float16),
                ],
                title='16bits',
                extra_args=dict(
                    dtype=np.uint16
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_quat_dataset(n, dtype=np.float32),
                ],
                title='32bits',
                extra_args=dict(
                    dtype=np.uint32
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_quat_dataset(n, dtype=np.float64),
                ],
                title='64bits',
                extra_args=dict(
                    dtype=np.uint64
                )
            )
        ],
        dataset_sizes=[2 ** n for n in range(25)],
        kernels=[
            Kernel(
                stmt="quaternion.encode_quat_to_uint(DATASET, dtype=EXTRA_ARGS['dtype'], encoder=fp.encode_fp_to_std_snorm)",
                setup='from fpq import quaternion\nfrom fpq import fp',
                label='encode_quat_to_uint(std)'
            ),
            Kernel(
                stmt="quaternion.encode_quat_to_uint(DATASET, dtype=EXTRA_ARGS['dtype'], encoder=fp.encode_fp_to_ogl_snorm)",
                setup='from fpq import quaternion\nfrom fpq import fp',
                label='encode_quat_to_uint(ogl)'
            ),
        ],
        xlabel='dataset sizes',
        title='encode_quat_to_uint'
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

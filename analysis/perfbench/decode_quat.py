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
from fpq import fp
from fpq import quaternion


def make_dataset(n, *, src_dtype, dst_dtype, encoder):
    dataset = make_quat_dataset(n, dtype=src_dtype)
    return quaternion.encode_quat_to_uint(dataset, dtype=dst_dtype, encoder=encoder)


def main():
    bm = Benchmark(
        datasets=[
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float16, dst_dtype=np.uint16, encoder=fp.encode_fp_to_std_snorm),
                    lambda n: make_dataset(n, src_dtype=np.float16, dst_dtype=np.uint16, encoder=fp.encode_fp_to_ogl_snorm)
                ],
                title='16bits',
                extra_args=dict(
                    dtype=np.float16
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float32, dst_dtype=np.uint32, encoder=fp.encode_fp_to_std_snorm),
                    lambda n: make_dataset(n, src_dtype=np.float32, dst_dtype=np.uint32, encoder=fp.encode_fp_to_ogl_snorm)
                ],
                title='32bits',
                extra_args=dict(
                    dtype=np.float32
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float64, dst_dtype=np.uint64, encoder=fp.encode_fp_to_std_snorm),
                    lambda n: make_dataset(n, src_dtype=np.float64, dst_dtype=np.uint64, encoder=fp.encode_fp_to_ogl_snorm)
                ],
                title='64bits',
                extra_args=dict(
                    dtype=np.float64
                )
            )
        ],
        dataset_sizes=[2 ** n for n in range(25)],
        kernels=[
            Kernel(
                stmt="quaternion.decode_uint_to_quat(DATASET, dtype=EXTRA_ARGS['dtype'], decoder=fp.decode_std_snorm_to_fp)",
                setup='from fpq import quaternion\nfrom fpq import fp',
                label='decode_uint_to_quat(std)'
            ),
            Kernel(
                stmt="quaternion.decode_uint_to_quat(DATASET, dtype=EXTRA_ARGS['dtype'], decoder=fp.decode_ogl_snorm_to_fp)",
                setup='from fpq import quaternion\nfrom fpq import fp',
                label='decode_uint_to_quat(ogl)'
            ),
        ],
        xlabel='dataset sizes',
        title='decode_uint_to_quat'
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

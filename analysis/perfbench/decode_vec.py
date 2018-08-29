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
from fpq import fp
from fpq import vector


def make_dataset(n, *, src_dtype, dst_dtype, nbits, encoder):
    dataset = make_vec_dataset(n, dtype=src_dtype)
    return vector.encode_vec_to_uint(dataset, dtype=dst_dtype, nbits=nbits, encoder=encoder)


def main():
    bm = Benchmark(
        datasets=[
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float16, dst_dtype=np.uint16, nbits=4, encoder=fp.encode_fp_to_std_snorm),
                    lambda n: make_dataset(n, src_dtype=np.float16, dst_dtype=np.uint16, nbits=4, encoder=fp.encode_fp_to_ogl_snorm)
                ],
                title='16bits',
                extra_args=dict(
                    dtype=np.float16,
                    nbits=4
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float32, dst_dtype=np.uint32, nbits=10, encoder=fp.encode_fp_to_std_snorm),
                    lambda n: make_dataset(n, src_dtype=np.float32, dst_dtype=np.uint32, nbits=10, encoder=fp.encode_fp_to_ogl_snorm)
                ],
                title='32bits',
                extra_args=dict(
                    dtype=np.float32,
                    nbits=10
                )
            ),
            Dataset(
                factories=[
                    lambda n: make_dataset(n, src_dtype=np.float64, dst_dtype=np.uint64, nbits=20, encoder=fp.encode_fp_to_std_snorm),
                    lambda n: make_dataset(n, src_dtype=np.float64, dst_dtype=np.uint64, nbits=20, encoder=fp.encode_fp_to_ogl_snorm)
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
                stmt="vector.decode_uint_to_vec(DATASET, dtype=EXTRA_ARGS['dtype'], nbits=EXTRA_ARGS['nbits'], decoder=fp.decode_std_snorm_to_fp)",
                setup='from fpq import vector\nfrom fpq import fp',
                label='decode_uint_to_vec(std)'
            ),
            Kernel(
                stmt="vector.decode_uint_to_vec(DATASET, dtype=EXTRA_ARGS['dtype'], nbits=EXTRA_ARGS['nbits'], decoder=fp.decode_ogl_snorm_to_fp)",
                setup='from fpq import vector\nfrom fpq import fp',
                label='decode_uint_to_vec(ogl)'
            ),
        ],
        xlabel='dataset sizes',
        title='decode_uint_to_vec'
    )
    bm.run()
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    bm.save_as_html(filepath=script_name + '.html')


if __name__ == '__main__':
    main()

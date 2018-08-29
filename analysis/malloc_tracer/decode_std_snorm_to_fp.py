#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import malloc_tracer as mt
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../../')
from data import make_snorm_dataset
from fpq import fp


def main():
    dataset_size = 100000
    src_dtype = np.float64
    dst_dtype = np.uint64
    nbits = 20

    dataset = fp.encode_fp_to_std_snorm(
        x=make_snorm_dataset(dataset_size, dtype=src_dtype),
        dtype=dst_dtype,
        nbits=nbits
    )

    mt.trace(
        fp.decode_std_snorm_to_fp,
        target_args=dict(
            x=dataset,
            dtype=src_dtype,
            nbits=nbits
        ),
        related_traces_output_mode=mt.RelatedTracesOutputMode.FOR_EACH_FILE,
        include_patterns={'*/fpq/*'},
        exclude_patterns=None
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import malloc_tracer as mt
import os
import sys
import math
import random
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../../')
from data import make_vec_dataset
from fpq import vector


def main():
    dataset_size = 100000
    src_dtype = np.float64
    dst_dtype = np.uint64
    nbits = 20

    dataset = vector.encode_vec_to_uint(
        make_vec_dataset(dataset_size, dtype=src_dtype),
        dtype=dst_dtype,
        nbits=nbits
    )

    mt.trace(
        vector.decode_uint_to_vec,
        target_args=dict(
            v=dataset,
            dtype=src_dtype,
            nbits=nbits
        ),
        related_traces_output_mode=mt.RelatedTracesOutputMode.FOR_EACH_FILE,
        include_patterns={'*/fpq/*'},
        exclude_patterns=None
    )


if __name__ == '__main__':
    main()

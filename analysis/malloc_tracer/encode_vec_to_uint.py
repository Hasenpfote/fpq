#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import malloc_tracer as mt
import math
import random
import os
import sys
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

    dataset = make_vec_dataset(dataset_size, dtype=src_dtype)

    mt.trace(
        vector.encode_vec_to_uint,
        target_args=dict(
            v=dataset,
            dtype=dst_dtype,
            nbits=nbits
        ),
        related_traces_output_mode=mt.RelatedTracesOutputMode.FOR_EACH_FILE,
        include_patterns={'*/fpq/*'},
        exclude_patterns=None
    )


if __name__ == '__main__':
    main()

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
from data import make_quat_dataset
from fpq import quaternion


def main():
    dataset_size = 100000
    src_dtype = np.float64
    dst_dtype = np.uint64

    dataset = make_quat_dataset(dataset_size, dtype=src_dtype)

    mt.trace(
        quaternion.encode_quat_to_uint,
        target_args=dict(
            q=dataset,
            dtype=dst_dtype,
        ),
        related_traces_output_mode=mt.RelatedTracesOutputMode.FOR_EACH_FILE,
        include_patterns={'*/fpq/*'},
        exclude_patterns=None
    )


if __name__ == '__main__':
    main()

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

    dataset = quaternion.encode_quat_to_uint(
        q=make_quat_dataset(dataset_size, dtype=src_dtype),
        dtype=dst_dtype
    )

    mt.trace(
        quaternion.decode_uint_to_quat,
        target_args=dict(
            q=dataset,
            dtype=src_dtype,
        ),
        related_traces_output_mode=mt.RelatedTracesOutputMode.FOR_EACH_FILE,
        include_patterns={'*/fpq/*'},
        exclude_patterns=None
    )


if __name__ == '__main__':
    main()

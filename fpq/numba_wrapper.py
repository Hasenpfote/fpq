#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
import numpy as np


try:
    from numba import jit
except ImportError:
    import warnings
    warning_text = \
        '\n\n' + '!' * 79 + '\n' + \
        'Could not import from numba.\n' + \
        'If numba is not installed, performance can be degraded in some functions.' + \
        '\n' + '!' * 79 + '\n'
    warnings.warn(warning_text)

    def _identity_decorator(*args, **kwargs):
        import types
        if (len(args) == 1) and isinstance(args[0], types.FunctionType):
            return args[0]

        def wrapper(fn):
            return fn

        return wrapper

    jit = _identity_decorator


def autocast(index_or_function=0):
    import types
    if isinstance(index_or_function, types.FunctionType):
        func = index_or_function
        index = 0
    else:
        func = None
        index = index_or_function

    def decorator(fn):
        def wrapper(*args, **kwargs):
            ret = fn(*args, **kwargs)
            if isinstance(ret, np.ndarray) or isinstance(ret, np.float16):
                return ret
            else:
                return args[index].dtype.type(ret)

        return wrapper

    return decorator if func is None else decorator(func)

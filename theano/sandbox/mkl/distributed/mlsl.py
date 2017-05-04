from ctypes import c_void_p, cdll
import numpy as np
from ctypes import *  # noqa

"""
Module contains MLSL-based transport implementation
for multinode communication Theano Ops; used in multinode.py
"""

# Library
libmlsl = cdll.LoadLibrary("libmlsl_theano.so")
# MLSL constants
null = c_void_p(0)
mlsl_float = 1
mlsl_double = 2

# MLSL prototypes
mlsl_init = libmlsl.mlsl_init
mlsl_finalize = libmlsl.mlsl_finalize
mlsl_set_global_batch_size = libmlsl.mlsl_set_global_batch_size
mlsl_set_param_count = libmlsl.mlsl_set_param_count
mlsl_create_distribution = libmlsl.mlsl_create_distribution
mlsl_create_operation = libmlsl.mlsl_create_operation
mlsl_rank = libmlsl.mlsl_rank
mlsl_size = libmlsl.mlsl_size
mlsl_start = libmlsl.mlsl_start
mlsl_wait = libmlsl.mlsl_wait


def addr(x):
    xaddr, offset = x.ctypes.data_as(c_void_p), 0
    for i in range(len(x.shape)):
        if x.strides[i] < 0:
            offset += (x.shape[i] - 1) * x.strides[i]
    xaddr.value += offset
    return xaddr


# dist_init()
# Initialize transport environment
# Returns my rank and total number of ranks
def dist_init():
    mlsl_init(null, null)
    return mlsl_rank(), mlsl_size()

dist_indexes = []
op_indexes = []


def ctxt_init(dist_idx):
    if dist_idx not in dist_indexes:
        # print 'ctxt_init: ' + str(dist_idx)
        mlsl_create_distribution(dist_idx)
        dist_indexes.append(dist_idx)
        return dist_idx


def set_global_batch_size(size):
    mlsl_set_global_batch_size(size)


def set_param_count(param_count):
    mlsl_set_param_count(param_count)


def coll_perform(blocking, inbuf, dtype, shapes, op_idx):
    # Adjust datatype
    if dtype == 'float32':
        mlsl_dtype = mlsl_float
    else:
        mlsl_dtype = mlsl_double

    # create MLSL operation only once
    if op_idx not in op_indexes:
        incount = np.prod(shapes)
        mlsl_create_operation(op_idx, incount, mlsl_dtype)
        op_indexes.append(op_idx)

    if blocking:  # or not "start" in is_start:
        pass
    else:
        mlsl_start(op_idx, inbuf)


def wait_perform(op_idx):
    mlsl_wait(op_idx)

"""
Optimizations addressing the convolution for mkl
"""
from __future__ import absolute_import, print_function, division
# import theano
# from theano import compile, gof
from theano.compile import optdb
# from theano.gof import local_optimizer
# from theano.gof.opt import copy_stack_trace

# from theano.tensor.opt import register_specialize_device
# from theano.tensor import TensorType
# from theano.tensor import opt

# from theano.tensor import nnet

from theano.gof import (local_optimizer, Optimizer, toolbox)
# from theano.gof.opt import LocalMetaOptimizer

from theano.sandbox.mkl import mkl_optimizer, register_opt, mkl_seqopt

from theano.sandbox.mkl.mkl_dummy import dummy_op

# global OPT
optdb.register('mkl_opt',
               mkl_seqopt,
               optdb.__position__.get('add_destroy_handler', 49.5) - 1,
               'mkl')

# local OPT
mkl_seqopt.register('mkl_local_optimizations', mkl_optimizer, 47.5,
                    'fast_run', 'fast_compile', 'mkl')

# show how global OPT works in here.
# TODO: write the real code soon


class Cut_I2U_U2I(Optimizer):
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        print("Intel Theano global opt test: Cut_I2U_U2I ")


mkl_seqopt.register('Cut_I2U_U2I', Cut_I2U_U2I(),
                    59,
                    'fast_run',
                    'fast_compile',
                    'mkl')  # TODO: how to make it mandatory for gpu_seqopt?


# Local OPT
@register_opt()
@local_optimizer([dummy_op])
def local_dummy(node):
    print("Intel Theano local OPT: local_dummy")
    return False

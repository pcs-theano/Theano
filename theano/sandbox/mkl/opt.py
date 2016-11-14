"""
Optimizations addressing the convolution for mkl
"""
from __future__ import absolute_import, print_function, division
# import theano
# from theano import compile, gof
from theano.compile import optdb
# from theano.gof import local_optimizer
from theano.gof.opt import copy_stack_trace

# from theano.tensor.opt import register_specialize_device
# from theano.tensor import TensorType
# from theano.tensor import opt

# from theano.tensor import nnet

from theano.gof import (local_optimizer, Optimizer, toolbox)
# from theano.gof.opt import LocalMetaOptimizer

from theano.sandbox.mkl import mkl_optimizer, register_opt, mkl_seqopt, mkl_available
from theano.sandbox.mkl.mkl_dummy import dummy_op

from theano.sandbox.mkl.basic_ops import (U2IGrad,
                                          I2U,
                                          I2UGrad,
                                          U2I_Pool,
                                          U2I_Relu,
                                          )
from theano.sandbox.mkl import mkl_relu
from theano.sandbox.mkl import mkl_pool

from theano.tensor.signal import pool
from theano.tensor.nnet.nnet import (Relu, ReluGrad)

# uniq_id for all the mkl Ops, to differentiate different layer even they've same parameters.
uniq_id = 0

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


@register_opt()
@local_optimizer([pool.Pool])
def local_pool_mkl(node):
    print ('@@@local_pool_mkl')
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, pool.Pool):
        return

    if not node.op.ignore_border:
        return

    x, ws, stride, pad = node.inputs
    if stride is None:
        stride = ws

    x_u2i = U2I_Pool(ignore_border=node.op.ignore_border,
                     mode=node.op.mode,
                     uniq_id=uniq_id)(x, ws, stride, pad)
    poolOut = mkl_pool.pool(ignore_border=node.op.ignore_border,
                            mode=node.op.mode,
                            uniq_id=uniq_id)(x_u2i, ws, stride, pad)
    z_i2u = I2U(uniq_id=uniq_id)(poolOut)

    rval = z_i2u
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@register_opt()
@local_optimizer([pool.MaxPoolGrad])
def local_poolGrad_mkl(node):
    print ('@@@local_poolGrad_mkl')
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, pool.MaxPoolGrad):
        return

    if not node.op.ignore_border:
        return

    x, maxout, gz, ws, stride, pad = node.inputs
    if stride is None:
        stride = ws

    x_u2i = U2I_Pool(ignore_border=node.op.ignore_border,
                     mode=node.op.mode,
                     uniq_id=uniq_id)(x, ws, stride, pad)
    gz_u2i = I2UGrad(uniq_id=uniq_id)(x_u2i, gz)
    poolGradOut = mkl_pool.poolGrad(ignore_border=node.op.ignore_border,
                                    mode=node.op.mode,
                                    uniq_id=uniq_id)(x_u2i, gz_u2i, ws, stride, pad)
    # gx_i2u = I2U(uniq_id=uniq_id)(poolGradOut)
    gx_i2u = U2IGrad(uniq_id=uniq_id)(x_u2i, poolGradOut)

    rval = gx_i2u
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@register_opt()
@local_optimizer([Relu])
def local_relu_mkl(node):
    print ('@@@local_relu_mkl')
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, Relu):
        return

    x, = node.inputs

    x_u2i = U2I_Relu(slope=node.op.slope, uniq_id=uniq_id)(x)
    reluOut = mkl_relu.Relu(slope=node.op.slope, uniq_id=uniq_id)(x_u2i)
    z_i2u = I2U(uniq_id=uniq_id)(reluOut)

    rval = z_i2u
    return [rval]


@register_opt()
@local_optimizer([ReluGrad])
def local_reluGrad_mkl(node):
    print ('@@@local_reluGrad_mkl')
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, ReluGrad):
        return

    x, gz = node.inputs

    x_u2i = U2I_Relu(slope=node.op.slope, uniq_id=uniq_id)(x)
    gz_u2i = I2UGrad(uniq_id=uniq_id)(x_u2i, gz)
    reluGradOut = mkl_relu.ReluGrad(slope=node.op.slope, uniq_id=uniq_id)(x_u2i, gz_u2i)
    gx_i2u = U2IGrad(uniq_id=uniq_id)(x_u2i, reluGradOut)

    rval = gx_i2u
    return [rval]

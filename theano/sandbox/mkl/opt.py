"""
Optimizations addressing the convolution for mkl
"""
from __future__ import absolute_import, print_function, division
import theano
from theano import compile, gof
from theano.compile import optdb
from theano.gof import local_optimizer
from theano.gof.opt import copy_stack_trace

from theano.sandbox.mkl.basic_ops import (U2I,
                                          U2IGrad,
                                          I2U,
                                          I2UGrad)
from theano.sandbox.mkl.mkl_conv import (conv_forward,
                                         conv_gradInputs,
                                         conv_gradWeights)
from theano.sandbox.mkl.mkl_relu import (Relu, ReluGrad)
from theano.sandbox.mkl.mkl_pool import (Pool, PoolGrad)

from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs)
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.opt import register_specialize_device
from theano.tensor import TensorType
from theano.tensor import opt

# mkl_conv opts
@local_optimizer([AbstractConv2d])
def local_abstractconv_mkl(node):
    if theano.config.cxx == "" or not theano.config.blas.ldflags:
        return None
    if not isinstance(node.op, AbstractConv2d):
        return None

    img, kern = node.inputs
    if not isinstance(img.type, TensorType) or \
       not isinstance(kern.type, TensorType):
        return None

    if node.op.imshp is None:
        ## can't use MKL function without imshp
        #conv_imshp = (None, None, None, None)
        return None
    else:
        conv_imshp = node.op.imshp

    if node.op.kshp is None:
        # Can't use MKL function without kshp
        #conv_kshp = (None, None, None, None)
        return None
    else:
        conv_kshp = node.op.kshp

    u2iOp = U2I('conv','dnnResourceSrc', tuple(conv_imshp) + 
                tuple(conv_kshp) + tuple(node.op.subsample) + 
                tuple(node.op.border_mode) + (1,), uniq_name = "U2I")
    img_converted = u2iOp(img)

    rval = conv_forward(imshp=conv_imshp,
                        kshp=conv_kshp,
                        border_mode=node.op.border_mode,
                        subsample=node.op.subsample)(img_converted, kern)

    i2uOp = I2U(uniq_name="I2U") 
    rval = i2uOp(rval)

    copy_stack_trace(node.outputs[0], rval)

    return [rval]

@local_optimizer([AbstractConv2d_gradWeights])
def local_abstractconv_gradweight_mkl(node):
    if theano.config.cxx == "" or not theano.config.blas.ldflags:
        return None
    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return None

    img, topgrad, shape = node.inputs
    if not isinstance(img.type, TensorType) or \
       not isinstance(topgrad.type, TensorType):
        return None

    if node.op.imshp is None:
        ## can't use MKL function without imshp
        #conv_imshp = (None, None, None, None)
        return None
    else:
        conv_imshp = node.op.imshp

    if node.op.kshp is None:
        # Can't use MKL function without kshp
        #conv_kshp = (None, None, None, None)
        return None
    else:
        conv_kshp = node.op.kshp

    rval = conv_gradWeights(imshp=conv_imshp,
                            kshp=conv_kshp,
                            border_mode=node.op.border_mode,
                            subsample=node.op.subsample)(img, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]


@local_optimizer([AbstractConv2d_gradInputs])
def local_abstractconv_gradinputs_mkl(node):
    if theano.config.cxx == "" or not theano.config.blas.ldflags:
        return None
    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, TensorType) or \
       not isinstance(topgrad.type, TensorType):
        return None

    if node.op.imshp is None:
        ## can't use MKL function without imshp
        #conv_imshp = (None, None, None, None)
        return None
    else:
        conv_imshp = node.op.imshp

    if node.op.kshp is None:
        # Can't use MKL function without kshp
        #conv_kshp = (None, None, None, None)
        return None
    else:
        conv_kshp = node.op.kshp

    rval = conv_gradInputs(imshp=conv_imshp,
                           kshp=conv_kshp,
                           border_mode=node.op.border_mode,
                           subsample=node.op.subsample)(kern, topgrad, shape)
    copy_stack_trace(node.outputs[0], rval)

    return [rval]

# Register Cpu Optmization
conv_groupopt = theano.gof.optdb.LocalGroupDB()
conv_groupopt.__name__ = "mkl_conv_opts"
register_specialize_device(conv_groupopt, 'fast_compile', 'fast_run', 'mkl')

conv_groupopt.register('local_abstractconv_mkl', local_abstractconv_mkl, 20,
                       'mkl', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv_gradweight_mkl',
                       local_abstractconv_gradweight_mkl, 20,
                       'mkl', 'fast_compile', 'fast_run')
conv_groupopt.register('local_abstractconv_gradinputs_mkl',
                       local_abstractconv_gradinputs_mkl, 20,
                       'mkl', 'fast_compile', 'fast_run')

relu_groupopt = theano.gof.optdb.LocalGroupDB()
relu_groupopt.__name__ = "mkl_relu_opts"
register_specialize_device(relu_groupopt, 'fast_compile', 'fast_run', 'mkl')

relu_groupopt.register('


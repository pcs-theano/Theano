from __future__ import absolute_import, print_function, division
import numpy
# Skip test if mkl is not available.
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises

import theano
from theano import config, tensor, scalar, gof
from theano.compile.ops import Shape_i
import theano.sandbox.mkl as mkl
from theano.tensor import TensorConstant, Elemwise, Alloc
from theano.tensor.signal import pool

from theano.sandbox.mkl.basic_ops import (U2IGrad,
                                          I2U,
                                          I2UGrad,
                                          U2I_Pool,
                                          U2I_Relu,
                                          )

if not mkl.mkl_available():
    raise SkipTest('Optional package MKL disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_mkl = theano.compile.mode.get_mode('FAST_RUN').including('mkl')
    mode_without_mkl = theano.compile.mode.get_mode('FAST_RUN').excluding('mkl')
else:
    mode_with_mkl = theano.compile.mode.get_default_mode().including('mkl')
    mode_without_mkl = theano.compile.mode.get_default_mode().excluding('mkl')


def test_mkl_add():
    a = tensor.matrix('a')
    b = tensor.matrix('b')

    z = a + b
    f = theano.function([a, b], z, mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()

    assert len(topo) == 1
    assert isinstance(topo[0].op, tensor.Elemwise)
    assert isinstance(topo[0].op.scalar_op, scalar.Add)

    vx = numpy.random.rand(5, 4).astype(theano.config.floatX)
    vy = numpy.random.rand(5, 4).astype(theano.config.floatX)

    assert numpy.all(f(vx, vy) == vx + vy)


def test_mkl_pool_forward():
    maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
    imval = numpy.random.rand(4, 2, 16, 16).astype(theano.config.floatX)
    if theano.config.floatX == 'float32':
        images = tensor.ftensor4()
    else:
        images = tensor.dtensor4()
    ignore_border = True
    mode = 'max'

    poolOut = pool.pool_2d(images, maxpoolshps[0], ignore_border, mode=mode)
    f = theano.function(inputs=[images], outputs=[poolOut], mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 3
    assert isinstance(topo[0].op, U2I_Pool)
    assert isinstance(topo[1].op, mkl.mkl_pool.pool)
    assert isinstance(topo[2].op, I2U)
    # U2I_Pool
    assert len(topo[0].inputs) == 4
    assert isinstance(topo[0].inputs[1], TensorConstant)
    assert isinstance(topo[0].inputs[3], TensorConstant)
    assert topo[0].inputs[1] == topo[0].inputs[2]
    # pool
    assert len(topo[1].inputs) == 4
    assert isinstance(topo[1].inputs[1], TensorConstant)
    assert isinstance(topo[1].inputs[3], TensorConstant)
    assert topo[1].inputs[1] == topo[1].inputs[2]
    assert topo[1].inputs[0].owner == topo[0]
    # I2U
    assert len(topo[2].inputs) == 1
    assert topo[2].inputs[0].owner == topo[1]
    # Output
    assert outputs[0].owner == topo[2]

    f1 = theano.function(inputs=[images, ], outputs=[poolOut, ], mode=mode_without_mkl)
    assert (numpy.asarray(f(imval)) == f1(imval)).all()


def test_mkl_pool_backward():

    predefineOps = [Shape_i, Shape_i, Shape_i, Shape_i, U2I_Pool,
                    Elemwise, Elemwise, Alloc, I2UGrad, pool.poolGrad, U2IGrad]

    maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
    imval = numpy.random.rand(4, 2, 16, 16).astype(theano.config.floatX)
    images = tensor.ftensor4()
    ignore_border = True
    mode = 'max'

    poolOut = pool.pool_2d(images, maxpoolshps[0], ignore_border, mode=mode)
    poolSum = tensor.sum(poolOut)
    poolBackward = tensor.grad(poolSum, [images])
    f = theano.function(inputs=[images, ], outputs=poolBackward, mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 11

    for i, node in enumerate(topo):
        assert isinstance(node.op, predefineOps[i])

    # U2I_Pool
    assert len(topo[4].inputs) == 4
    assert isinstance(topo[4].inputs[1], TensorConstant)
    assert isinstance(topo[4].inputs[3], TensorConstant)
    assert topo[4].inputs[1] == topo[4].inputs[2]
    # I2UGrad
    assert len(topo[8].inputs) == 2
    assert topo[8].inputs[0].owner == topo[4]
    assert topo[8].inputs[1].owner == topo[7]
    # poolGrad
    assert len(topo[9].inputs) == 5
    assert topo[9].inputs[0].owner == topo[4]
    assert topo[9].inputs[1].owner == topo[8]
    # U2IGrad
    assert len(topo[10].inputs) == 2
    assert topo[10].inputs[0].owner == topo[4]
    assert topo[10].inputs[1].owner == topo[9]
    # Output
    assert outputs[0].owner == topo[10]

    f1 = theano.function(inputs=[images, ], outputs=[poolOut, ], mode=mode_without_mkl)
    # assert (numpy.asarray(f(imval)) == f1(imval)).all()


def test_mkl_relu_forward():
    shape = (256, 96, 55, 55)
    if theano.config.floatX == 'float32':
        x = tensor.fmatrix('x')
    else:
        x = tensor.dmatrix('x')
    xx = x.reshape(shape)
    y = tensor.nnet.Relu(slope=1)(xx)

    yy = tensor.nnet.relu(xx)

    f = theano.function(inputs=[xx], outputs=y, mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 3

    # U2I_Relu
    assert len(topo[0].inputs) == 1
    assert isinstance(topo[0].op, U2I_Relu)
    assert topo[0].inputs[0] == inputs[0]
    # Relu
    assert isinstance(topo[1].op, mkl.mkl_relu.Relu)
    assert len(topo[1].inputs) == 1
    assert topo[1].inputs[0].owner == topo[0]
    # I2U
    assert isinstance(topo[2].op, I2U)
    assert len(topo[2].inputs) == 1
    assert topo[2].inputs[0].owner == topo[1]
    # output
    assert outputs[0].owner == topo[2]

    imval = numpy.random.rand(256, 96, 55, 55).astype(theano.config.floatX)

    f1 = theano.function(inputs=[xx], outputs=yy, mode=mode_without_mkl)
    #assert (numpy.asarray(f(imval)) == f1(imval)).all()


def test_mkl_relu_backward():
    pass


def test_mkl_pool_relu():
    """
    Test the combination graph of pooling and relu.
    :return:
    """
    shape = (256, 96, 55, 55)
    x = tensor.fmatrix('x')
    xx = x.reshape(shape)
    y = tensor.nnet.Relu(slope=1)(xx)
    maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
    imval = numpy.random.rand(4, 2, 16, 16).astype(theano.config.floatX)
    ignore_border = True
    mode = 'max'
    poolOut = pool.pool_2d(y, maxpoolshps[0], ignore_border, mode=mode)
    f = theano.function(inputs=[xx], outputs=[poolOut], mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 4

    # U2I_Relu
    assert len(topo[0].inputs) == 1
    assert isinstance(topo[0].op, U2I_Relu)
    assert topo[0].inputs[0] == inputs[0]
    # Relu
    assert isinstance(topo[1].op, mkl.mkl_relu.Relu)
    assert len(topo[1].inputs) == 1
    assert topo[1].inputs[0].owner == topo[0]
    # pool
    assert len(topo[2].inputs) == 4
    assert topo[2].inputs[0].owner == topo[1]
    # I2U
    assert isinstance(topo[3].op, I2U)
    assert len(topo[3].inputs) == 1
    assert topo[3].inputs[0].owner == topo[2]
    # output
    assert outputs[0].owner == topo[3]


if __name__ == '__main__':
    if mkl.mkl_available():
        test_mkl_add()
        test_mkl_pool_forward()
        test_mkl_relu_forward()
        test_mkl_pool_relu()
    else:
        print('Optional package MKL disabled')

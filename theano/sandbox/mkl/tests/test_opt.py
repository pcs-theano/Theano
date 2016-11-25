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
                                          U2IPool,
                                          U2IRelu,
                                          U2ILRN
                                          )

from theano.tensor.nnet.lrn import lrn

if not mkl.mkl_available():
    raise SkipTest('Optional package MKL disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_mkl = theano.compile.mode.get_mode('FAST_RUN').including('mkl')
    mode_without_mkl = theano.compile.mode.get_mode('FAST_RUN').excluding('mkl')
else:
    mode_with_mkl = theano.compile.mode.get_default_mode().including('mkl')
    mode_without_mkl = theano.compile.mode.get_default_mode().excluding('mkl')


def test_mkl_available():
    # TODO
    pass


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

    print('test_mkl_add() pass..')


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
    assert isinstance(topo[0].op, U2IPool)
    assert isinstance(topo[1].op, mkl.mkl_pool.Pool)
    assert isinstance(topo[2].op, I2U)
    # U2IPool
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

    print('test_mkl_pool_forward() pass..')


def test_mkl_pool_backward():

    predefineOps = [Shape_i, Shape_i, Shape_i, Shape_i, U2IPool,
                    Elemwise, Elemwise, mkl.mkl_pool.Pool, Alloc, I2UGrad, mkl.mkl_pool.PoolGrad, U2IGrad]

    maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
    imval = numpy.random.rand(4, 2, 16, 16).astype(theano.config.floatX)
    if theano.config.floatX == 'float32':
        images = tensor.ftensor4()
    else:
        images = tensor.dtensor4()
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
    assert len(topo) == 12

    for i, node in enumerate(topo):
        assert isinstance(node.op, predefineOps[i])

    # U2IPool
    assert len(topo[4].inputs) == 4
    assert isinstance(topo[4].inputs[1], TensorConstant)
    assert isinstance(topo[4].inputs[3], TensorConstant)
    assert topo[4].inputs[1] == topo[4].inputs[2]
    # I2UGrad
    assert len(topo[9].inputs) == 2
    assert topo[9].inputs[0].owner == topo[7]
    assert topo[9].inputs[1].owner == topo[8]
    # poolGrad
    assert len(topo[10].inputs) == 5
    assert topo[10].inputs[0].owner == topo[4]
    assert topo[10].inputs[1].owner == topo[9]
    # U2IGrad
    assert len(topo[11].inputs) == 2
    assert topo[11].inputs[1].owner == topo[10]
    # Output
    assert outputs[0].owner == topo[11]

    f1 = theano.function(inputs=[images, ], outputs=poolBackward, mode=mode_without_mkl)
    assert (numpy.asarray(f(imval)) == f1(imval)).all()

    print('test_mkl_pool_backward() pass..')


def test_mkl_relu_forward():
    shape = (256, 96, 55, 55)
    if theano.config.floatX == 'float32':
        x = tensor.ftensor4('x')
    else:
        x = tensor.dtensor4('x')
    y = tensor.nnet.AbstractRelu(slope=1)(x)

    yy = tensor.nnet.relu(x)

    f = theano.function(inputs=[x], outputs=y, mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 3

    # U2IRelu
    assert len(topo[0].inputs) == 1
    assert isinstance(topo[0].op, U2IRelu)
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

    f1 = theano.function(inputs=[x], outputs=yy, mode=mode_without_mkl)
    
    #assert numpy.all(f(imval) == f1(imval))
    print('test_mkl_relu_forward() pass..')


def test_mkl_relu_backward():
    predefineOps = [U2IRelu, mkl.mkl_relu.Relu, I2U, Shape_i, Shape_i, Shape_i, Shape_i, Alloc,
                    I2UGrad, mkl.mkl_relu.ReluGrad, U2IGrad]
    if theano.config.floatX == 'float32':
        x = tensor.ftensor4('x')
    else:
        x = tensor.dtensor4('x')

    y = tensor.nnet.relu(x)

    s = tensor.sum(y)
    z = tensor.grad(s, [x])
    f = theano.function([x], z, mode=mode_with_mkl)

    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 11

    for i, node in enumerate(topo):
        assert isinstance(node.op, predefineOps[i])

    imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)
    f(imval)
    print('test_mkl_relu_backward() pass..')


def test_mkl_pool_relu():
    """
    Test the combination graph of pooling and relu.
    :return:
    """
    x = tensor.ftensor4('x')
    y = tensor.nnet.AbstractRelu(slope=1)(x)
    maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
    imval = numpy.random.rand(4, 2, 16, 16).astype(theano.config.floatX)
    ignore_border = True
    mode = 'max'
    poolOut = pool.pool_2d(y, maxpoolshps[0], ignore_border, mode=mode)
    f = theano.function(inputs=[x], outputs=[poolOut], mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 4

    # U2I_Relu
    assert len(topo[0].inputs) == 1
    assert isinstance(topo[0].op, U2IRelu)
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

    print('test_mkl_pool_relu() pass..')


def test_mkl_lrn_forward():
    if theano.config.floatX == 'float32':
        x = tensor.ftensor4()
    else:
        x = tensor.dtensor4()

    y = lrn(x)

    f = theano.function([x], y, mode=mode_with_mkl)

    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 3

    assert isinstance(topo[0].op, U2ILRN)
    assert isinstance(topo[1].op, mkl.mkl_lrn.LRN)
    assert isinstance(topo[2].op, I2U)

    assert outputs[0].owner == topo[2]

    imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)
    f(imval)
    print('test_mkl_lrn_forward() pass..')


def test_mkl_lrn_backward():
    predefineOps = [U2ILRN, mkl.mkl_lrn.LRN, I2U, 
                    Shape_i, Shape_i, Shape_i, Shape_i, Alloc, I2UGrad, mkl.mkl_lrn.LRNGrad, U2IGrad]

    if theano.config.floatX == 'float32':
        x = tensor.ftensor4()
    else:
        x = tensor.dtensor4()

    y = lrn(x)
    z = tensor.grad(tensor.sum(y), [x])
    f = theano.function([x], z, mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs
    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 11

    for i, node in enumerate(topo):
        isinstance(node.op, predefineOps[i])

    # U2I_LRN
    assert len(topo[0].inputs) == 1
    assert topo[0].inputs[0] == inputs[0]

    # LRN
    assert len(topo[1].inputs) == 1
    assert topo[1].inputs[0].owner == topo[0]

    # I2UGrad
    assert len(topo[8].inputs) == 2
    assert topo[8].inputs[0].owner == topo[1]
    assert topo[8].inputs[1].owner == topo[7]

    # LRNGrad
    assert len(topo[9].inputs) == 2
    assert topo[9].inputs[0].owner == topo[0]
    assert topo[9].inputs[1].owner == topo[8]

    # U2IGrad
    assert len(topo[10].inputs) == 2
    assert topo[10].inputs[0] == inputs[0]
    assert topo[10].inputs[1].owner == topo[9]

    # Output
    assert outputs[0].owner == topo[10]

    imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)
    f(imval)
    print('test_mkl_lrn_backward() pass..')


def test_mkl_opt_with_wrong_dim():

    if theano.config.floatX == 'float32':
        x = tensor.ftensor3()
    else:
        x = tensor.dtensor3()

    imval = numpy.random.rand(3, 4, 5).astype(theano.config.floatX)
    
    try:
        y = tensor.nnet.relu(x)
        f = theano.function([x], y, mode=mode_with_mkl)

        topo = f.maker.fgraph.toposort()
        for node in topo:
            if isinstance(node.op, U2IRelu) or isinstance(node.op, I2U):
                raise ValueError('For 3D tensor, there should not have MKL OP in graph')
            else:
                f(imval)

    except Exception as e:
        raise Exception('nnet.relu(x) ' + str(e))
    
    try:
        z = pool.pool_2d(x, (2, 2), True, mode='max')
        f1 = theano.function([x], z, mode=mode_with_mkl)

        topo = f1.maker.fgraph.toposort()
        for node in topo:
            if isinstance(node.op, U2IPool) or isinstance(node.op, I2U):
                raise ValueError('For 3D tensor, there should not have MKL OP in graph')
            else:
                f1(imval)
    except Exception as e:
        raise Exception('pool.pool_2d(x) ' + str(e))

    print('test_mkl_opt_with_wrong_dim() pass..')


def test_mkl_3_relu_forward():

    if theano.config.floatX == 'float32':
        x = tensor.ftensor4()
    else:
        x = tensor.dtensor4()

    y = tensor.nnet.relu(x)
    z = tensor.nnet.relu(y)
    zz = tensor.nnet.relu(z)

    f = theano.function([x], zz, mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()

    assert len([node for node in topo if isinstance(node.op, mkl.mkl_relu.Relu)]) == 3

    for node in topo:
        if isinstance(node.op, I2U):
            assert not isinstance(node.inputs[0].owner.op, U2IRelu)
    imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)
    f(imval)
    print('test_mkl_3_relu_forward() pass..')


def test_mkl_3_relu_backward():
    predefineOPs = [U2IRelu, mkl.mkl_relu.Relu, mkl.mkl_relu.Relu, mkl.mkl_relu.Relu, I2U, 
                    Shape_i, Shape_i, Shape_i, Shape_i, Alloc, I2UGrad, 
                    mkl.mkl_relu.ReluGrad, mkl.mkl_relu.ReluGrad, mkl.mkl_relu.ReluGrad, U2IGrad]
    if theano.config.floatX == 'float32':
        x = tensor.ftensor4()
    else:
        x = tensor.dtensor4()

    y = tensor.nnet.relu(x)
    z = tensor.nnet.relu(y)
    zz = tensor.nnet.relu(z)

    s = tensor.sum(zz)
    t = tensor.grad(s, [x])

    f = theano.function([x], t, mode=mode_with_mkl)
    topo = f.maker.fgraph.toposort()
    inputs = f.maker.fgraph.inputs
    outputs = f.maker.fgraph.outputs

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert len(topo) == 15

    # U2IRelu
    assert len(topo[0].inputs) == 1
    assert topo[0].inputs[0] == inputs[0]
    # U2IGrad
    assert len(topo[14].inputs) == 2
    assert topo[14].inputs[0] == inputs[0]
    assert isinstance(topo[14].inputs[1].owner.op, mkl.mkl_relu.ReluGrad)
    # Output
    assert outputs[0].owner == topo[14]

    imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)
    f(imval)
    print('test_mkl_3_relu_backward() pass..')


if __name__ == '__main__':
    theano.config.floatX = 'float32'
    if mkl.mkl_available():
        test_mkl_add()
        test_mkl_pool_forward()
        test_mkl_pool_backward()
        test_mkl_relu_forward()
        test_mkl_relu_backward()
        test_mkl_pool_relu()
        test_mkl_lrn_forward()
        test_mkl_lrn_backward()
        # test_mkl_opt_with_wrong_dim()
        test_mkl_3_relu_forward()
        test_mkl_3_relu_backward()
    else:
        print('Optional package MKL disabled')


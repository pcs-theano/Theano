from __future__ import absolute_import, print_function, division

import theano
from theano import gof
from theano.tensor import basic as tensor


class AbstractLRN(gof.Op):
    __props__ = ('slope', 'alpha', 'beta', 'k', 'n')

    def __init__(self, slope=1, alpha=1e-4, beta=0.75, k=2, n=5):
        self.alpha = alpha
        self.beta  = beta
        self.k     = k
        self.n     = n
        self.slope = slope

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()
        x = tensor.as_tensor_variable(x)
        return gof.Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [AbstractLRNGrad(slope=self.slope, 
                                alpha=self.alpha,
                                beta=self.beta,
                                k=self.k,
                                n=self.n)(x, gz)]

    def perform(self, node, inp, out_):
        x, = inp
        z, = out_


class AbstractLRNGrad(gof.Op):
    __props__ = ('slope', 'alpha', 'beta', 'k', 'n')

    def __init__(self, slope=1, alpha=1e-4, beta=0.75, k=2, n=5, fp='default.txt'):
        self.slope = slope
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n
        self.fp = fp

    def make_node(self, x, gz):
        if x.type.ndim != 4:
            raise TypeError()
        x = tensor.as_tensor_variable(x)
        return gof.Apply(self, [x, gz], [x.type()])

    def perform(self, node, inp, out_):
        x, gz = inp
        gx, = out_
        gx = gz


def lrn(x, slope=1, alpha=1e-4, beta=0.75, k=2, n=5):
    if theano.sandbox.mkl.mkl_available.avail is None:
        theano.sandbox.mkl.mkl_available()

    if (theano.sandbox.mkl.mkl_available.avail is True) and (x.type.ndim == 4):
        return AbstractLRN(slope, alpha, beta, k, n)(x)
    else:
        # TODO: need a numpy implement
        raise NotImplementedError('LRN: MKL not available or dimension is wrong.')


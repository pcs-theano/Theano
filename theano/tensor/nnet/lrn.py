from __future__ import absolute_import, print_function, division

import theano
from theano import gof
from theano.tensor import basic as tensor


class LRN(gof.Op):
    __props__ = ('slope',)

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
        return [LRNGrad(slope=self.slope)(x, gz)]

    def perform(self, node, inp, out_):
        x, = inp
        z, = out_



class LRNGrad(gof.Op):
    __props__ = ('slope',)

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


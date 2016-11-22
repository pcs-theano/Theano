

import theano
from theano import tensor as T

import theano.sandbox.mkl
import theano.tensor.nnet
import theano.tensor.nnet.lrn
from theano.tensor.nnet.lrn import LRN

import numpy

x = T.ftensor4()
y = LRN()(x)

# forward
theano.printing.pydotprint(y, outfile='lrn_fwd_before.png', var_with_name_simple=True)
f = theano.function([x], y)
theano.printing.pydotprint(f, outfile='lrn_fwd_after.png', var_with_name_simple=True)

# backward
z = T.grad(T.sum(y), [x])
theano.printing.pydotprint(z, outfile='lrn_bwd_before.png', var_with_name_simple=True)
f1 = theano.function([x], z)
theano.printing.pydotprint(f1, outfile='lrn_bwd_after.png', var_with_name_simple=True)

# random
imval = numpy.random.rand(4, 2, 4, 4).astype(numpy.float32)
f(imval)
f1(imval)

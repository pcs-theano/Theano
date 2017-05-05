from __future__ import absolute_import, print_function, division

import numpy
from nose.plugins.skip import SkipTest

import theano
import theano.tensor as T
from theano.sandbox import mkl

if not mkl.mkl_available:
    raise SkipTest('Optional package MKL disabled')

try:
    import theano.sandbox.mlsl.distributed as distributed
except ImportError as e:
    print ('Failed to import mkl.distributed module, please double check')

dist = distributed.Distribution()
print ('dist.rank: ', dist.rank)
print ('dist.size: ', dist.size)

distributed.set_global_batch_size(2)
distributed.set_param_count(1)

shape = (1, 1, 5, 5)
base_array = numpy.ones(shape, dtype=numpy.float32)
input = T.ftensor4('input')

input_array = base_array * (dist.rank + 1)
print ('input_array:', input_array)

out = distributed.allreduce(input)

f = theano.function([theano.compile.io.In(input, mutable=True, borrow=True)],
                    theano.compile.io.Out(out, borrow=True))
result = f(input_array)

print ('## rank: %d: ' % dist.rank)
print ('result:', result)

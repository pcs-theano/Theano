import theano
import numpy as np
from theano import tensor as T

shape = (256, 96, 55, 55)
x = T.fmatrix('x')
y = T.fmatrix('y')
xx = x.reshape(shape)
y = T.nnet.relu(xx)

theano.printing.pydotprint(y, outfile="relu_before.png", var_with_name_simple=True)
f = theano.function(inputs=[xx], outputs=y)
theano.printing.pydotprint(f, outfile="relu_after.png", var_with_name_simple=True)

input_x = np.random.rand(shape[0], shape[1], shape[2], shape[3]).astype(theano.config.floatX)
z = f(input_x)

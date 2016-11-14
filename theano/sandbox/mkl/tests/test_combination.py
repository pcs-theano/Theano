import theano
import numpy as np
from theano import tensor as T
from theano.tensor.signal import pool
import theano.sandbox.mkl

shape = (256, 96, 55, 55)
x = T.fmatrix('x')
y = T.fmatrix('y')
xx = x.reshape(shape)
# y = T.nnet.relu(xx)
y = T.nnet.Relu(slope=1)(xx)

maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
imval = np.random.rand(4, 2, 16, 16)

ignore_border = True
mode = 'max'
direction = 'forward'
# direction = 'backward'

poolOut = pool.pool_2d(y, maxpoolshps[0], ignore_border, mode=mode)
if direction == 'forward':
    theano.printing.pydotprint(poolOut, outfile="relu_pool_before_opt.png", var_with_name_simple=True)
    f = theano.function(inputs=[xx, ], outputs=[poolOut, ])
    theano.printing.pydotprint(f, outfile="relu_pool_after_opt.png", var_with_name_simple=True)
elif direction == 'backward':
    poolSum = T.sum(poolOut)
    poolBackward = T.grad(poolSum, [xx])
    theano.printing.pydotprint(poolBackward, outfile="relu_poolBackward_before_opt.png", var_with_name_simple=True)
    f2 = theano.function(inputs=[xx, ], outputs=poolBackward)
    theano.printing.pydotprint(f2, outfile="relu_poolBackward_after_opt.png", var_with_name_simple=True)
else:
    print ("Invalid direction, only forward or backward allowed!")

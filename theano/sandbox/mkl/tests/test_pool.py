import theano
import numpy as np
from theano import tensor as T
from theano.tensor.signal import pool
import theano.sandbox.mkl

maxpoolshps = ((1, 1), (2, 2), (3, 3), (2, 3))
imval = np.random.rand(4, 2, 16, 16)
images = T.ftensor4()
ignore_border = True
mode = 'max'
direction = 'forward'
# direction = 'backward'

poolOut = pool.pool_2d(images, maxpoolshps[0], ignore_border, mode=mode)
if direction == 'forward':
    theano.printing.pydotprint(poolOut, outfile="pool_before_opt.png", var_with_name_simple=True)
    f = theano.function(inputs=[images, ], outputs=[poolOut, ])
    theano.printing.pydotprint(f, outfile="pool_after_opt.png", var_with_name_simple=True)
elif direction == 'backward':
    poolSum = T.sum(poolOut)
    poolBackward = T.grad(poolSum, [images])
    theano.printing.pydotprint(poolBackward, outfile="poolBackward_before_opt.png", var_with_name_simple=True)
    f2 = theano.function(inputs=[images, ], outputs=poolBackward)
    theano.printing.pydotprint(f2, outfile="poolBackward_after_opt.png", var_with_name_simple=True)
else:
    print ("Invalid direction, only forward or backward allowed!")

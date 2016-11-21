import theano
from theano import tensor as T
import numpy as np


def run_test(direction = 'forward'):
    print ('=' * 60)
    print ('generate relu graph before and after opt for %s pass' % direction)
    x = T.ftensor4('x')
    y = T.nnet.relu(x)

    imval = np.random.rand(4, 2, 4, 4).astype(np.float32)

    if direction == 'forward':
        theano.printing.pydotprint(y, outfile="relu_before_opt.png", var_with_name_simple=True)
        f = theano.function(inputs=[x], outputs=y)
        theano.printing.pydotprint(f, outfile="relu_after_opt.png", var_with_name_simple=True)
        f(imval)
    elif direction == 'backward':
        reluSum = T.sum(y)
        reluBackward = T.grad(reluSum, [x])
        theano.printing.pydotprint(reluBackward, outfile="reluGrad_before_opt.png", var_with_name_simple=True)
        f = theano.function(inputs=[x], outputs=reluBackward)
        theano.printing.pydotprint(f, outfile="reluGrad_after_opt.png", var_with_name_simple=True)
        f(imval)
    else:
        print ("Invalid direction, only forward or backward allowed!")

if __name__ == '__main__':
    run_test('forward')
    run_test('backward')

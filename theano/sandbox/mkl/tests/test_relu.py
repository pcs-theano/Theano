import theano
from theano import tensor as T

shape = (256, 96, 55, 55)
x = T.fmatrix('x')
y = T.fmatrix('y')
xx = x.reshape(shape)
# y = T.nnet.relu(xx)
y = T.nnet.Relu(slope=1)(xx)

direction = 'forward'
# direction = 'backward'

if direction == 'forward':
    theano.printing.pydotprint(y, outfile="relu_before.png", var_with_name_simple=True)
    f = theano.function(inputs=[xx], outputs=y)
    theano.printing.pydotprint(f, outfile="relu_after.png", var_with_name_simple=True)
elif direction == 'backward':
    reluSum = T.sum(y)
    reluBackward = T.grad(reluSum, [xx])
    theano.printing.pydotprint(reluBackward, outfile="reluGrad_before.png", var_with_name_simple=True)
    f = theano.function(inputs=[xx], outputs=reluBackward)
    theano.printing.pydotprint(f, outfile="reluGrad_after.png", var_with_name_simple=True)
else:
    print ("Invalid direction, only forward or backward allowed!")

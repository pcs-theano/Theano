import sys
import theano

import theano.tensor as T
import numpy as np

from baselayers import SoftmaxLayer, DropoutLayer, DataLayer, Conv2DLayer, MaxPool2DLayer, LocalResponseNormalization2DLayer, ConcatLayer, GlobalPoolLayer, DenseLayer, AveragePool2DLayer, FlattenLayer

def aux_tower(input_layer, input_shape):
    
    net = {}
    params = []
    weight_types = []
        
    # input shape = (14x14x512or528)
    net['aux_tower_pool'] = AveragePool2DLayer(
    input_layer, input_shape, pool_size=5, stride=3, pad=0, ignore_border=False)
    output_shape = net['aux_tower_pool'].get_output_shape_for()
    
    # output shape = (4x4x512or528)
    filter_shape = (128, output_shape[1], 1, 1)
    net['aux_tower_1x1'] = Conv2DLayer(net['aux_tower_pool'].output,
    output_shape, filter_shape, flip_filters=False, init_b=0.2)
    params +=  [net['aux_tower_1x1'].W, net['aux_tower_1x1'].b]
    weight_types += net['aux_tower_1x1'].weight_types
    output_shape = net['aux_tower_1x1'].get_output_shape_for()
    
    # output shape = (2048)
    net['FC_1'] = DenseLayer(net['aux_tower_1x1'].output, output_shape, 1024,
    init_b=0.2)
    params +=  [net['FC_1'].W, net['FC_1'].b]
    weight_types += net['FC_1'].weight_types
    output_shape = net['FC_1'].get_output_shape_for()
         
    #drp = Dropout(input=fc.output,n_in=1024, n_out=1024, prob_drop=0.7)
    net['aux_dropout'] = DropoutLayer(net['FC_1'].output, output_shape, prob_drop=0.7)
    output_shape = net['aux_dropout'].get_output_shape_for() 

    net['classifier'] = SoftmaxLayer(net['aux_dropout'].output, output_shape, 1000)
    params += net['classifier'].params
    weight_types += net['classifier'].weight_types
               
    return params, weight_types, net['classifier'].p_y_given_x, net['classifier'].negative_log_likelihood
        
def build_inception_module(input_layer, input_shape, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    params = []
    weight_types = []
    input_shapes = [1,1,1,1]
    
    filter_shape = (nfilters[1], input_shape[1], 1, 1)
    net['1x1'] = Conv2DLayer(input_layer, input_shape, filter_shape,
    flip_filters=False, init_b=0.2)
    params +=  [net['1x1'].W, net['1x1'].b]
    weight_types += net['1x1'].weight_types
    output_shape = net['1x1'].get_output_shape_for()
    #print ('1x1 '+str(output_shape))
    input_shapes[0] = output_shape
   
    filter_shape = (nfilters[2], input_shape[1], 1, 1)
    net['3x3_reduce'] = Conv2DLayer(
        input_layer, input_shape, filter_shape, flip_filters=False, init_b=0.2)
    params += [net['3x3_reduce'].W, net['3x3_reduce'].b]
    weight_types += net['3x3_reduce'].weight_types
    output_shape = net['3x3_reduce'].get_output_shape_for()
    
    filter_shape = (nfilters[3], output_shape[1], 3, 3) 
    net['3x3'] = Conv2DLayer(
        net['3x3_reduce'].output, output_shape, filter_shape, padsize=1,
        flip_filters=False, init_b=0.2)
    params += [net['3x3'].W, net['3x3'].b]
    weight_types += net['3x3'].weight_types
    output_shape = net['3x3'].get_output_shape_for()
    #print ('3x3 '+str(output_shape))
    input_shapes[1] = output_shape

    filter_shape = (nfilters[4], input_shape[1], 1, 1)
    net['5x5_reduce'] = Conv2DLayer(
        input_layer, input_shape, filter_shape, flip_filters=False, init_b=0.2)
    params += [net['5x5_reduce'].W, net['5x5_reduce'].b]
    weight_types += net['5x5_reduce'].weight_types
    output_shape = net['5x5_reduce'].get_output_shape_for()
    
    filter_shape = (nfilters[5], output_shape[1], 5, 5)
    net['5x5'] = Conv2DLayer(
        net['5x5_reduce'].output, output_shape, filter_shape, padsize=2,
        flip_filters=False, init_b=0.2)
    params += [net['5x5'].W, net['5x5'].b]
    weight_types += net['5x5'].weight_types
    output_shape = net['5x5'].get_output_shape_for()
    #print ('5x5 '+str(output_shape))
    input_shapes[2] = output_shape
    
    net['pool'] = MaxPool2DLayer(input_layer, input_shape, pool_size=3, stride=1, pad=1)
    output_shape = net['pool'].get_output_shape_for()
    
    filter_shape = (nfilters[0], output_shape[1], 1, 1)
    net['pool_proj'] = Conv2DLayer(
        net['pool'].output, output_shape, filter_shape, flip_filters=False,
        init_b=0.2)    
    params += [net['pool_proj'].W, net['pool_proj'].b]
    weight_types += net['pool_proj'].weight_types
    output_shape = net['pool_proj'].get_output_shape_for()
    #print ('pool_proj '+str(output_shape))
    input_shapes[3] = output_shape
    #print(input_shapes)
    
    net['output'] = ConcatLayer([
        net['1x1'].output,
        net['3x3'].output,
        net['5x5'].output,
        net['pool_proj'].output,
        ], input_shapes)

    return net['output'], params, weight_types

class googlenet(object):
    
    def __init__(self, input_shape):
        
        self.input_shape = input_shape
        x = T.tensor4('x', dtype=theano.config.floatX)
        y = T.vector('y', dtype='int32')
        net = {}
        params = []
        weight_types = []
        
        net['input'] = x
        
        filter_shape = (64, input_shape[1], 7, 7)
        net['conv1/7x7_s2'] = Conv2DLayer(
        net['input'], input_shape, filter_shape, convstride=2, padsize=3,
        flip_filters=False, init_b=0.2)
        params += [net['conv1/7x7_s2'].W, net['conv1/7x7_s2'].b]
        weight_types += net['conv1/7x7_s2'].weight_types
        output_shape = net['conv1/7x7_s2'].get_output_shape_for()
        
        net['pool1/3x3_s2'] = MaxPool2DLayer(
        net['conv1/7x7_s2'].output, output_shape, pool_size=3, stride=2, ignore_border=False)
        output_shape = net['pool1/3x3_s2'].get_output_shape_for()
        
        net['pool1/norm1'] = LocalResponseNormalization2DLayer(net['pool1/3x3_s2'].output, output_shape, alpha=0.00002, k=1)
        output_shape = net['pool1/norm1'].get_output_shape_for()
        
        filter_shape = (64, output_shape[1], 1, 1)
        net['conv2/3x3_reduce'] = Conv2DLayer(
        net['pool1/norm1'].output, output_shape, filter_shape,
        flip_filters=False, init_b=0.2)
        params += [net['conv2/3x3_reduce'].W, net['conv2/3x3_reduce'].b]
        weight_types += net['conv2/3x3_reduce'].weight_types
        output_shape = net['conv2/3x3_reduce'].get_output_shape_for()
        
        filter_shape = (192, output_shape[1], 3, 3)
        net['conv2/3x3'] = Conv2DLayer(
        net['conv2/3x3_reduce'].output, output_shape, filter_shape, padsize=1,
        flip_filters=False, init_b=0.2)
        params += [net['conv2/3x3'].W, net['conv2/3x3'].b]
        weight_types += net['conv2/3x3'].weight_types
        output_shape = net['conv2/3x3'].get_output_shape_for()
        
        net['conv2/norm2'] = LocalResponseNormalization2DLayer(net['conv2/3x3'].output, output_shape, alpha=0.00002, k=1)
        output_shape = net['conv2/norm2'].get_output_shape_for()
        
        net['pool2/3x3_s2'] = MaxPool2DLayer(
        net['conv2/norm2'].output, output_shape,  pool_size=3, stride=2, ignore_border=False)
        output_shape = net['pool2/3x3_s2'].get_output_shape_for()
        
        net['inception_3a'], pre_params, pre_weight_types = build_inception_module(net['pool2/3x3_s2'].output, output_shape, [32, 64, 96, 128, 16, 32])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_3a'].get_output_shape_for()
        
        net['inception_3b'], pre_params, pre_weight_types = build_inception_module(net['inception_3a'].output, output_shape, [64, 128, 128, 192, 32, 96])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_3b'].get_output_shape_for()
        
        net['pool3/3x3_s2'] = MaxPool2DLayer(
        net['inception_3b'].output, output_shape, pool_size=3, stride=2, ignore_border=False)
        output_shape = net['pool3/3x3_s2'].get_output_shape_for()
        
        net['inception_4a'], pre_params, pre_weight_types = build_inception_module(net['pool3/3x3_s2'].output, output_shape, [64, 192, 96, 208, 16, 48])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4a'].get_output_shape_for()
        aux_tower_shape1 = output_shape
        
        pre_params, pre_weight_types, p_y_given_x_1, negative_log_likelihood_1 = aux_tower(net['inception_4a'].output, aux_tower_shape1)
        params += pre_params
        weight_types += pre_weight_types

        net['inception_4b'], pre_params, pre_weight_types = build_inception_module(net['inception_4a'].output, output_shape, [64, 160, 112, 224, 24, 64])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4b'].get_output_shape_for()
        
        net['inception_4c'], pre_params, pre_weight_types = build_inception_module(net['inception_4b'].output, output_shape, [64, 128, 128, 256, 24, 64])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4c'].get_output_shape_for()
        
        net['inception_4d'], pre_params, pre_weight_types = build_inception_module(net['inception_4c'].output, output_shape, [64, 112, 144, 288, 32, 64])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4d'].get_output_shape_for()
        aux_tower_shape2 = output_shape
        
        pre_params, pre_weight_types, p_y_given_x_2, negative_log_likelihood_2 = aux_tower(net['inception_4d'].output, aux_tower_shape2)
        params += pre_params
        weight_types += pre_weight_types

        net['inception_4e'], pre_params, pre_weight_types = build_inception_module(net['inception_4d'].output, output_shape, [128, 256, 160, 320, 32, 128])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4e'].get_output_shape_for()
        
        net['pool4/3x3_s2'] = MaxPool2DLayer(
        net['inception_4e'].output, output_shape, pool_size=3, stride=2, ignore_border=False)
        output_shape = net['pool4/3x3_s2'].get_output_shape_for()
        
        net['inception_5a'], pre_params, pre_weight_types = build_inception_module(net['pool4/3x3_s2'].output, output_shape, [128, 256, 160, 320, 32, 128])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_5a'].get_output_shape_for()
        
        net['inception_5b'], pre_params, pre_weight_types = build_inception_module(net['inception_5a'].output, output_shape, [128, 384, 192, 384, 48, 128])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_5b'].get_output_shape_for()
        
        
#        net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b'].output, output_shape)
#        output_shape = net['pool5/7x7_s1'].get_output_shape_for()
        
        net['pool5/7x7_s1'] = AveragePool2DLayer(
        net['inception_5b'].output, output_shape, pool_size=7, stride=1, pad=0, ignore_border=False)
        output_shape = net['pool5/7x7_s1'].get_output_shape_for()
        
        net['flatten'] = FlattenLayer(net['pool5/7x7_s1'].output, output_shape)
        output_shape = net['flatten'].get_output_shape_for()

        net['dropout'] = DropoutLayer(net['flatten'].output, output_shape, prob_drop=0.4)
        output_shape = net['dropout'].get_output_shape_for() 
        
        net['loss3/classifier'] = SoftmaxLayer(net['dropout'].output, output_shape, 1000)
        params += net['loss3/classifier'].params
        weight_types += net['loss3/classifier'].weight_types
        
        #### aux_tower classifier
        self.net = net
        self.params = params
        self.weight_types = weight_types
        
        #self.cost = net['loss3/classifier'].categorical_crossentropy(y)
        self.cost = net['loss3/classifier'].negative_log_likelihood(y) + 0.3*negative_log_likelihood_1(y) + 0.3*negative_log_likelihood_2(y)
        self.errors = net['loss3/classifier'].errors(y)
        self.errors_top_5 = net['loss3/classifier'].errors_top_x(y, 5)
        self.x = x
        self.y = y
        
    def set_dropout_off(self):
        DropoutLayer.SetDropoutOff()
    
    def set_dropout_on(self):
        DropoutLayer.SetDropoutOn()


def compile_val_model(model, batch_size=256, image_size=(3, 224, 224)):

    input_shape = (batch_size,) + image_size
    
    cost = model.cost
    x = model.x
    y = model.y
    errors = model.errors
    errors_top_5 = model.errors_top_5

    shared_x = theano.shared(np.zeros(input_shape, dtype=theano.config.floatX),borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype='int32'), borrow=True)

    validate_outputs = [cost, errors, errors_top_5]

    validate_model = theano.function([], validate_outputs,
                                     givens=[(x, shared_x), (y, shared_y)])

    return (validate_model,
            shared_x, shared_y)
            
def compile_train_model(model, learning_rate=0.01, batch_size=256, image_size=(3, 224, 224), use_momentum=True, momentum=0.9, weight_decay=0.0002):

    input_shape = (batch_size,) + image_size
    
    x = model.x
    y = model.y
    weight_types = model.weight_types
    cost = model.cost
    params = model.params
    errors = model.errors
    errors_top_5 = model.errors_top_5


    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(learning_rate))
    lr = T.scalar('lr', dtype=theano.config.floatX)  # symbolic learning rate

    shared_x = theano.shared(np.zeros(input_shape, dtype=theano.config.floatX),borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype='int32'), borrow=True)
                             
    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]

    if use_momentum:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):
            if weight_type == 'W':
                real_grad = grad_i + weight_decay * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")
            vel_i_next = momentum * vel_i - real_lr * real_grad
            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))
    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):
            if weight_type == 'W':
                updates.append((param_i,
                                param_i - lr * grad_i - weight_decay * lr * param_i))
            elif weight_type == 'b':
                updates.append((param_i, param_i - 2 * lr * grad_i))
            else:
                raise TypeError("Weight Type Error")

    #Define Theano Functions

    train_model = theano.function([], cost, updates=updates, 
                                  givens=[(x, shared_x), (y, shared_y), (lr, learning_rate)])

    train_error = theano.function(
        [], errors, givens=[(x, shared_x), (y, shared_y)])

    return (train_model, train_error,
            shared_x, shared_y, learning_rate)

def set_learning_rate(shared_lr, iter):
    if (iter+1) % 320000 == 0:
        temp = shared_lr.get_value()*0.96
        temp = np.float32(temp)
        shared_lr.set_value(temp)


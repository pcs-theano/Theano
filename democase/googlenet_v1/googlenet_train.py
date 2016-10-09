import theano
import sys
import os
import time
import theano.tensor as T
import numpy as np
import cPickle as pickle
from googlenet_theano import googlenet, compile_models, set_learning_rate
from params import get_params, load_net_state, load_params, save_net_state
from read_lmdb import read_lmdb

def googlenet_train(batch_size=32, image_size=(3, 224, 224), n_epochs=60, mkldnn=False):

    train_lmdb_path = '/path/to/your/imagenet/ilsvrc2012/ilsvrc12_train_lmdb'
    val_lmdb_path = '/path/to/your/imagenet/ilsvrc2012/ilsvrc12_val_lmdb'

    input_shape = (batch_size,) + image_size
    model = googlenet(input_shape, mkldnn)

    ##### get training and validation data from lmdb file
    train_lmdb_iterator = read_lmdb(batch_size, train_lmdb_path)
    train_data_size = train_lmdb_iterator.total_number
    n_train_batches = train_data_size / batch_size
    print('n_train_batches = '+ str(n_train_batches))

    val_lmdb_iterator = read_lmdb(batch_size, val_lmdb_path)
    val_data_size = val_lmdb_iterator.total_number
    n_val_batches = val_data_size / batch_size
    print('n_val_batches = '+ str(n_val_batches))
    
    ## COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error,
        shared_x, shared_y, shared_lr, vels) = compile_models(model, batch_size = batch_size)

    ####load net state
    start_epoch = 0
    net_params = load_net_state(start_epoch)
    if net_params:
        load_params(model.params, net_params['model_params'])
        load_params(vels, net_params['vels'])
        epoch = net_params['epoch']
        minibatch_index = net_params['minibatch_index']
        train_lmdb_iterator.set_cursor(net_params['minibatch_index'])
    else:
        epoch = 0
        minibatch_index = 0

    print('... training')
    # stroe history cost, and print the average cost in a frequency of 40 iterations
    cost_array = []
    while(epoch < n_epochs):
        count = 0
        while(minibatch_index < n_train_batches):
            count = count + 1
            if count == 1:
                s = time.time()
            if count == 20:
                e = time.time()
                print "time per 20 iter:", (e - s)

            ####training
            idx = epoch * n_train_batches + minibatch_index
            train_data, train_label = train_lmdb_iterator.next()
            shared_x.set_value(train_data)
            shared_y.set_value(train_label)
            set_learning_rate(shared_lr, idx)

            cost_ij = train_model()
            error_ij = train_error()

            cost_array.append(cost_ij)
            if idx % 40 == 0:
                average_cost = np.array(cost_array).mean()
                cost_array = []
                print('iter %d, cost %f, error_ij %f' % (idx, average_cost, error_ij)) #FIXME, mean loss for googlenet
            minibatch_index += 1
            if minibatch_index == n_train_batches:
                minibatch_index = 0

        ###validation
        val_top5_errors = []
        val_top1_errors = []
        model.set_dropout_off()

        for validation_index in xrange(0, n_val_batches):
            val_data, val_label = val_lmdb_iterator.next()
            shared_x.set_value(val_data)
            shared_y.set_value(val_label)
            cost, errors, errors_top_5 = validate_model()
            val_top5_errors.append(errors_top_5)
            val_top1_errors.append(errors)
        model.set_dropout_on()
        val_top5_err = np.mean(val_top5_errors)
        val_top1_err = np.mean(val_top1_errors)
        print('epoch %i, validation error (top1) %f %%, (top5) %f %%' %
              (epoch, val_top5_err * 100., val_top1_err * 100.))

        ###save params every epoch
        net_params['model_params'] = get_params(model.params)
        net_params['vels'] = get_params(vels)
        net_params['minibatch_index'] = minibatch_index
        net_params['epoch'] = epoch
        save_net_state(net_params, epoch)

        epoch = epoch + 1

if __name__ =='__main__':
    googlenet_train(batch_size=32,mkldnn=True)

import theano
theano.config.floatX='float32'
import sys
import os
import time
import timeit
import theano.tensor as T
import numpy as np
import cPickle as pickle
from googlenet_theano import googlenet, compile_val_model, compile_train_model, set_learning_rate
from params import get_params, load_net_state, load_params, save_net_state, save_figure, common_save, common_load
from read_lmdb import read_lmdb
import math

def googlenet_train(train_batch_size=32, val_batch_size=50, image_size=(3, 224, 224), n_epochs=60):

    #mean_path = '/home/2T/caffe/data/ilsvrc12/imagenet_mean.binaryproto'
    train_lmdb_path = '/home/2T/imagenet/ilsvrc2012/lmdb/ilsvrc12_train_lmdb'
    val_lmdb_path = '/home/2T/imagenet/ilsvrc2012/lmdb/ilsvrc12_val_lmdb'

    train_input_shape = (train_batch_size,) + image_size
    val_input_shape = (val_batch_size,) + image_size
    trainning_model = googlenet(train_input_shape)
    validating_model = googlenet(val_input_shape)

    #####read lmdb
    train_lmdb_iterator = read_lmdb(train_batch_size, train_lmdb_path)
    train_data_size = train_lmdb_iterator.total_number
    n_train_batches = train_data_size / train_batch_size
    print('n_train_batches = '+ str(n_train_batches))

    val_lmdb_iterator = read_lmdb(val_batch_size, val_lmdb_path)
    val_data_size = val_lmdb_iterator.total_number
    n_val_batches = val_data_size / val_batch_size
    print('n_val_batches = '+ str(n_val_batches))
    
    ## COMPILE FUNCTIONS ##
    (train_model, train_error,
        train_shared_x, train_shared_y, shared_lr) = compile_train_model(trainning_model, batch_size=train_batch_size)

    (val_model, val_shared_x, val_shared_y) = compile_val_model(validating_model, batch_size=val_batch_size)
    
    all_costs = []
    all_errors = []

    ####load net state
    net_params = load_net_state()
    if net_params:
        load_params(model.params, net_params['model_params'])
        train_lmdb_iterator.set_cursor(net_params['minibatch_index'])
        all_errors = net_params['all_errors']
        all_costs = net_params['all_costs']
        epoch = net_params['epoch']
        minibatch_index = net_params['minibatch_index']
    else:
        all_costs = []
        all_errors = []
        epoch = 0
        minibatch_index = 0

    print('... training')
    while(epoch < n_epochs):

        while(minibatch_index < n_train_batches):
            ####training
            #print(minibatch_index)
            iter = epoch * n_train_batches + minibatch_index
            print('training @ epoch = %d : iter = %d : totoal_batches = %d' %(epoch, iter, n_train_batches))
            begin_time = time.time()
            train_data, train_label = train_lmdb_iterator.next()
            train_shared_x.set_value(train_data)
            train_shared_y.set_value(train_label)
            set_learning_rate(shared_lr, iter)

            #begin_time = time.time()
            cost_ij = train_model()
            error_ij = train_error()
            all_costs.append(cost_ij)
            all_errors.append(error_ij)
            print('train_error: %f %%' %(error_ij*100))
            print('trian_cost: %f' %(cost_ij))
            end_time = time.time()
            print('Time per iteration: %f' % (end_time - begin_time))
            if math.isnan(cost_ij):
                nan_params = get_params(model.params)
                common_save(nan_params, './nan_params')
                sys.exit(0)

            ###validation		 
            if (iter+1) % (4*n_train_batches) == 0:
                values = get_params(trainning_model.params)
                load_params(validating_model.params, values)

                validation_erorrs = []
                validating_model.set_dropout_off()

                for validation_index in xrange(0, n_val_batches):
                    #print('validation_index = %d : total_batches = %d' %(validation_index, n_val_batches))
                    val_data, val_label = val_lmdb_iterator.next()
                    val_shared_x.set_value(val_data)
                    val_shared_y.set_value(val_label)
                    cost, errors, errors_top_5 = val_model()
                    validation_erorrs.append(errors_top_5)
                validating_model.set_dropout_on()
                this_validation_error = np.mean(validation_erorrs)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_error * 100.))

            ###save params every epoch
            if (iter+1) % n_train_batches == 0:
                net_params['model_params'] = get_params(model.params)
                net_params['minibatch_index'] = minibatch_index
                net_params['all_costs'] = all_costs
                net_params['all_errors'] = all_errors
                net_params['epoch'] = epoch
                save_net_state(net_params)
                save_figure(all_costs, all_errors)
                
            minibatch_index += 1

        if minibatch_index == n_train_batches:
            minibatch_index = 0
        epoch = epoch + 1


if __name__ =='__main__':
    googlenet_train(train_batch_size=32, val_batch_size=50)

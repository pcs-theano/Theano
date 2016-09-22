import theano
theano.config.floatX='float32'
import theano.tensor as T
import numpy as np
from googlenet_theano import googlenet, compile_val_model, compile_train_model, set_learning_rate
import time
from datetime import datetime

def time_theano_run(func, info_string):
    num_batches = 100
    num_steps_burn_in = 10
    durations = []
    for i in xrange(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = func()
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: Iteration %d, %s, time: %.2f ms' %
                      (datetime.now(), i - num_steps_burn_in, info_string, duration*1000))
            durations.append(duration)
    durations = np.array(durations)
    print('%s: Average %s pass: %.2f ms ' %
          (datetime.now(), info_string, durations.mean()*1000))


def googlenet_train(train_batch_size=32, val_batch_size=50, image_size=(3, 224, 224), n_epochs=60):
    
    # train model
    train_input_shape = (train_batch_size,) + image_size
    training_model = googlenet(train_input_shape)

    (train_model, train_error, train_shared_x, train_shared_y,
            shared_lr) = compile_train_model(training_model, batch_size=train_batch_size)
    train_images = np.random.random_integers(0, 255, train_input_shape).astype('float32')
    train_labels = np.random.random_integers(0, 999, train_batch_size).astype('int32')
    train_shared_x.set_value(train_images)
    train_shared_y.set_value(train_labels)

    iter = 0
    set_learning_rate(shared_lr, iter)

    # validation model 
    val_input_shape = (val_batch_size,) + image_size
    validation_model = googlenet(val_input_shape)

    (val_model, val_shared_x, val_shared_y) = compile_val_model(validation_model, batch_size=val_batch_size)
    val_images = np.random.random_integers(0, 255, val_input_shape).astype('float32')
    val_labels = np.random.random_integers(0, 999, val_batch_size).astype('int32')
    val_shared_x.set_value(val_images)
    val_shared_y.set_value(val_labels)

    # forward benchmark
    validation_model.set_dropout_off()
    time_theano_run(val_model, 'Forward')

    # forward-backward benchmark
    training_model.set_dropout_on()
    time_theano_run(train_model, 'Forward-Backward')

if __name__ =='__main__':
    googlenet_train(train_batch_size=32, val_batch_size=50)

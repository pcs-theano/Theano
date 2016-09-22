import cPickle as pickle
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_params(params):
    result = []
    for param in params:
	result.append(param.get_value())
    return result

def load_params(params, values):
    print('params length = %d' %(len(params)))
    print('caffe_params length = %d' %(len(values)))
    if len(params) != len(values):
        raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(values), len(params)))
    for i in xrange(0,len(params)):
        if params[i].get_value().shape != values[i].shape:
            print (i,' : ', params[i].get_value().shape,values[i].shape)
            raise ValueError("mismatch: parameter has shape %r but value to "
                             "set has shape %r" %
                             (params[i].get_value().shape,values[i].shape))
        else:
            params[i].set_value(values[i])
    print('Params loaded sucessfully!')

def save_figure(costs, errors):
    if os.path.exists('./costs.jpg'):
        os.remove('costs.jpg')
    if os.path.exists('./errors.jpg'):
        os.remove('errors.jpg')

    fig1 = plt.figure('fig1')
    plt.plot(np.array(costs))
    plt.savefig('./costs.jpg')
    fig2 = plt.figure('fig2')
    plt.plot(np.array(errors))
    plt.savefig('./errors.jpg')

def common_save(param, file_name):
     if os.path.exists(file_name):
        os.remove(file_name)

     with file(file_name, "wb") as f:
        pickle.dump(param, f, -1)
        f.close()
        print('Param saved sucessfully!')
   
def common_load(file_name):
    with open(file_name, 'rb') as f:
        result = pickle.load(f)
        print('Param loaded sucessfully!')
    return result
 
def load_net_state():
    if os.path.exists('./net_state.pkl'):
        with file('./net_state.pkl', "rb") as f:
            net_state = pickle.load(f)
            f.close()
            print('Net_state loaded sucessfully!')
            return net_state
    else:
        return {}

def save_net_state(net_params):
     if os.path.exists('./net_state.pkl'):
        os.remove('./net_state.pkl')

     with file('./net_state.pkl', "wb") as f:
        pickle.dump(net_params, f, -1)
        f.close()
        print('Net_params saved sucessfully!')

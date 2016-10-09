import cPickle as pickle
import numpy as np
import os

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
    for p, v in zip(params, values):
        if p.get_value().shape != v.shape:
            raise ValueError("mismatch: parameter has shape %r but value to "
                             "set has shape %r" %
                             (p.get_value().shape, v.shape))
        else:
            p.set_value(v)
    print('Params loaded sucessfully!')

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
 
def load_net_state(epoch):
    para_file = 'params_saved/net_state_'+str(epoch)+'.pkl'
    print('loading para from file %s ' % para_file)
    if os.path.exists(para_file):
        with file(para_file, "rb") as f:
            net_state = pickle.load(f)
            f.close()
            print('Net_state loaded sucessfully!')
	    return net_state
    else:
	return {}

def save_net_state(net_params, epoch):
    if not os.path.exists('params_saved'):
        os.mkdir('params_saved')
    with file('params_saved/net_state_'+str(epoch)+'.pkl', "wb") as f:
        pickle.dump(net_params, f, -1)
        f.close()
        print('Net_params saved sucessfully!')

"""
multinode.py
Module contains transport independent implementation of communication Ops for Theano
Supported transports:
    - mlsl (requires mlsl.py)
"""
from ctypes import c_void_p
import numpy as np
import theano
from theano.gof import local_optimizer
from theano.tensor.opt import in2out
from theano.compile.mode import optdb

import mlsl  # MLSL library dependent code
from mlsl import (dist_init,
                  ctxt_init,
                  coll_perform,
                  wait_perform)

seqn = 0


class Distribution(object):
    """
    Distribution class
    Initializes environment, own rank and number of ranks (size)

    Dependency (must be defined in the transport layer):
    dist_init()
    """
    init = 0
    rank = 0
    size = 1
    if not init:
        init = 1
        rank, size = dist_init()


def set_global_batch_size(size):
    print 'set_global_batch_size ' + str(size)
    mlsl.set_global_batch_size(size)


def set_param_count(param_count):
    print 'set_param_count ' + str(param_count)
    mlsl.set_param_count(param_count)


def addr(x):
    xaddr, offset = x.ctypes.data_as(c_void_p), 0
    for i in range(len(x.shape)):
        if x.strides[i] < 0:
            offset += (x.shape[i] - 1) * x.strides[i]
    xaddr.value += offset
    return xaddr


class AllReduce(Distribution, theano.Op):
    __props__ = ('blocking', 'inplace', 'seqn')

    def __init__(self, blocking=True, inplace=False, seqn=0):
        self.blocking = blocking
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}
        self.seqn = seqn
        if (seqn > 0):
            self.ctxt = ctxt_init(self.seqn)

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        outputs = [x.type(), ]
        # Non-blocking call return an extra int64 output as a placeholder for additional info
        if not self.blocking:
            y = theano.tensor.lscalar()
            outputs.append(y.type())
        return theano.Apply(self, [x], outputs)

    def perform(self, node, inputs, outputs):
        x, = inputs
        inbuf = addr(x)

        if self.blocking:
            y, = outputs
            req_addr = 0
        else:
            # Non-blocking call return an extra output - integer reqeust
            y, req = outputs
            req[0] = np.empty((), dtype=np.int64)
            req_addr = addr(req[0])  # noqa

        if self.inplace:
            y[0] = x
            outbuf = addr(x)  # noqa
        else:
            y[0] = np.empty(x.shape, dtype=x.dtype)

        # MLSL does in-place communication, so no extra buffer for output
        coll_perform(self.blocking, inbuf, x.dtype, x.shape, self.seqn)

        if not self.blocking:
            wait_perform(self.seqn)

    def grad(self, inputs, grads):
        return [coll(grads[0], kind='Allreduce', blocking=self.blocking)]


def allreduce(x, blocking=True, inplace=False):
    global seqn
    seqn += 1

    if blocking:
        return AllReduce(blocking=blocking, inplace=inplace)(x)
    else:
        return AllReduce(blocking=blocking, inplace=inplace, seqn=seqn)(x)


@local_optimizer([AllReduce()], inplace=True)
def local_allreduce_inplace(node):
    op = node.op
    if isinstance(op, AllReduce) and op.inplace is False:
        outputs = AllReduce(blocking=op.blocking,
                            inplace=True,
                            seqn=op.seqn)(*node.inputs)
        if isinstance(outputs, list):
            return outputs
        else:
            return [outputs]

allreduce_inplace = in2out(local_allreduce_inplace, name="allreduce_inplace")
optdb.register('allreduce inplace',
               allreduce_inplace,
               100.0, 'fast_run', 'inplace', 'allreduce_inplace')

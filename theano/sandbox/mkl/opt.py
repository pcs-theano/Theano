"""
Optimizations addressing the convolution for mkl
"""
from __future__ import absolute_import, print_function, division
import logging
import theano
from theano import gof, tensor, scalar
from theano.compile import optdb
from theano.gof import (local_optimizer, Optimizer, toolbox)
from theano.sandbox.mkl import mkl_optimizer, register_opt, mkl_seqopt, mkl_available

from theano.sandbox.mkl.basic_ops import (U2IGrad,
                                          I2U,
                                          I2UGrad,
                                          U2IPool,
                                          U2IRelu,
                                          U2ILRN,
                                          U2IConv,
                                          U2IElemwiseSum,
                                          U2IBatchNormalization,
                                          )
from theano.sandbox.mkl import mkl_relu
from theano.sandbox.mkl import mkl_pool
from theano.sandbox.mkl import mkl_lrn, mkl_conv, mkl_elemwise, mkl_bn

from theano.tensor.signal import pool
from theano.tensor.nnet.nnet import (AbstractRelu, AbstractReluGrad)

from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs)

_logger = logging.getLogger('theano.sandbox.mkl.opt')

# uniq_id for all the mkl Ops, to differentiate different layer even they've same parameters.
uniq_id = 0

# global OPT
optdb.register('mkl_opt',
               mkl_seqopt,
               0.09,
               'mkl')

# local OPT
mkl_seqopt.register('mkl_local_optimizations', mkl_optimizer, 20,
                    'fast_run', 'fast_compile', 'mkl')


class CutMKLDataConversionChain(Optimizer):
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        list_forward = ['Relu', 'Pool', 'Conv2D', 'LRN', 'ElemwiseSum']
        list_i2u = ['I2U']
        list_u2i = ['U2IPool', 'U2IRelu', 'U2IConv', 'U2IElemwiseSum', 'U2ILRN']
        list_backward = ['ReluGrad', 'PoolGrad', 'ConvGradInputs', 'ConvGradWeights', 'LRNGrad', 'ElemwiseSum']
        list_i2u_back = ['I2UGrad', 'U2IElemwiseSum']
        list_u2i_back = ['U2IGrad', 'I2U']
        for node in fgraph.toposort():
            # backward
            if node.op.__class__.__name__ in list_i2u_back:
                out = node.outputs[0]
                for inp in node.inputs:
                    if isinstance(inp.owner, gof.Apply) and inp.owner.op.__class__.__name__ in list_u2i_back:
                        for inpOP in list(inp.owner.inputs):
                            if isinstance(inpOP.owner, gof.Apply) and inpOP.owner.op.__class__.__name__ in list_backward:
                                fgraph.replace_validate(out, inpOP.owner.outputs[0])
            # forward
            if node.op.__class__.__name__ in list_u2i:
                out = node.outputs[0]
                for inp in node.inputs:
                    if isinstance(inp.owner, gof.Apply) and inp.owner.op.__class__.__name__ in list_i2u:
                        for inpOP in list(inp.owner.inputs):
                            if isinstance(inpOP.owner, gof.Apply) and inpOP.owner.op.__class__.__name__ in list_forward:
                                fgraph.replace_validate(out, inpOP.owner.outputs[0])


mkl_seqopt.register('CutMKLDataConversionChain', CutMKLDataConversionChain(),
                    40,
                    'fast_run',
                    'fast_compile',
                    'mkl')


# Global Optimizer for replace 'conv() + bias' with conv_with_bias()
class ReplaceConvBias(Optimizer):
    def __init__(self):
        super(ReplaceConvBias, self).__init__()

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def _check_add_bias_(self, node):
        out = node.outputs
        if (isinstance(out[0].clients[0][0].op, tensor.Elemwise) and
                isinstance(out[0].clients[0][0].op.scalar_op, scalar.Add)):
            if len(out[0].clients[0][0].inputs) == 2:
                if out[0].clients[0][0].inputs[0] is out[0]:
                    bias = out[0].clients[0][0].inputs[1]
                else:
                    bias = out[0].clients[0][0].inputs[0]
                # Get DimShuffle node
                bias_owner = bias.owner
                if bias_owner is None:
                    return bias
                elif isinstance(bias_owner.op, tensor.DimShuffle) and (bias_owner.inputs[0].owner is None):
                    return bias_owner.inputs[0]
                else:
                    return None

        return None

    def _check_grad_bias_(self, node, i):
        if isinstance(node.op, tensor.Elemwise):
            assert i == 0
            assert len(node.outputs[0].clients) >= 3
        elif isinstance(node.op, tensor.Split):
            assert len(node.outputs[i].clients) >= 3

        op = []
        pre_op = [tensor.DimShuffle, tensor.Elemwise, tensor.DimShuffle]
        for c in node.outputs[i].clients:
            if isinstance(c[0].op, tensor.Sum):
                c_ = c[0]
                for i in range(3):
                    if hasattr(c_.outputs[0], 'clients'):
                        op.append(getattr(c_.outputs[0].clients[0][0], 'op', None))
                        c_ = c_.outputs[0].clients[0][0]
                    else:
                        op.append(None)

                if all([isinstance(op[i], pre_op[i]) for i in range(3)]):
                    return c_.outputs[0]

        return None

    def apply(self, fgraph):
        global uniq_id
        did_something = True
        while did_something:
            did_something = False
            topo = fgraph.toposort()
            for node in topo:
                if (node in fgraph.apply_nodes) and isinstance(node.op, AbstractConv2d):
                    inp = node.inputs
                    out = node.outputs
                    imshp = getattr(node.op, 'imshp', None)
                    kshp = getattr(node.op, 'kshp', None)
                    border_mode = getattr(node.op, 'border_mode', 'valid')
                    subsample = getattr(node.op, 'subsample', (1, 1))
                    filter_flip = getattr(node.op, 'filter_flip', False)
                    filter_dilation = getattr(node.op, 'filter_dilation', (1, 1))

                    # Get Elemwise node
                    if (len(out) == 1 and (not out[0] in fgraph.outputs) and
                            isinstance(out[0].clients[0][0].op, tensor.Elemwise) and
                            isinstance(out[0].clients[0][0].op.scalar_op, scalar.Add)):
                        if len(out[0].clients[0][0].inputs) == 2:
                            if out[0].clients[0][0].inputs[0] is out[0]:
                                bias = out[0].clients[0][0].inputs[1]
                            else:
                                bias = out[0].clients[0][0].inputs[0]
                            # Get DimShuffle node
                            bias_owner = bias.owner
                            if (bias_owner is None):
                                try:
                                    uniq_id += 1
                                    inp_0 = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample,
                                                    filter_dilation=filter_dilation, uniq_id=uniq_id)(inp[0])
                                    uniq_id += 1
                                    out_0 = mkl_conv.Conv2D(imshp=imshp,
                                                            kshp=kshp,
                                                            border_mode=border_mode,
                                                            subsample=subsample,
                                                            filter_flip=filter_flip,
                                                            filter_dilation=filter_dilation,
                                                            uniq_id=uniq_id)(image=inp_0, weight=inp[1], bias=bias)
                                    fgraph.repalce_validate(out[0].clients[0][0].outputs[0],
                                                            out_0,
                                                            'ReplaceConvBias')
                                    did_something = True
                                except Exception as e:
                                    raise
                            elif isinstance(bias_owner.op, tensor.DimShuffle) and (bias_owner.inputs[0].owner is None):
                                try:
                                    uniq_id += 1
                                    inp_0 = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample,
                                                    filter_dilation=filter_dilation, uniq_id=uniq_id)(inp[0])
                                    uniq_id += 1
                                    out_0 = mkl_conv.Conv2D(imshp=imshp,
                                                            kshp=kshp,
                                                            border_mode=border_mode,
                                                            subsample=subsample,
                                                            filter_flip=filter_flip,
                                                            filter_dilation=filter_dilation,
                                                            uniq_id=uniq_id)(image=inp_0, weight=inp[1], bias=bias_owner.inputs[0])
                                    uniq_id += 1
                                    out_1 = I2U(uniq_id=uniq_id)(out_0)
                                    fgraph.replace_validate(out[0].clients[0][0].outputs[0],
                                                            out_1,
                                                            'ReplaceConvBias')
                                    # theano.printing.pydotprint(fgraph, outfile='replace_conv_fw.png', var_with_name_simple=True)
                                    did_something = True
                                except Exception as e:
                                    raise
                            else:
                                pass
                elif (node in fgraph.apply_nodes) and isinstance(node.op, AbstractConv2d_gradWeights):
                    inp = node.inputs  # 0-image, 1-gz, 2-shape
                    out = node.outputs
                    imshp = getattr(node.op, 'imshp', None)
                    kshp = getattr(node.op, 'kshp', None)
                    border_mode = getattr(node.op, 'border_mode', 'valid')
                    subsample = getattr(node.op, 'subsample', (1, 1))
                    filter_flip = getattr(node.op, 'filter_flip', False)
                    filter_dilation = getattr(node.op, 'filter_dilation', (1, 1))

                    assert len(inp) == 3 and len(out) == 1
                    for i, c in enumerate(inp[0].clients):
                        if hasattr(c[0], 'op') and isinstance(c[0].op, U2IConv):
                            for cc in c[0].outputs[0].clients:
                                if isinstance(cc[0].op, mkl_conv.Conv2D) and len(cc[0].inputs) == 3:
                                    # theano.printing.pydotprint(fgraph, outfile='zzzz.png', var_with_name_simple=True)
                                    weight, bias = cc[0].inputs[1:3]
                                    try:
                                        uniq_id += 1
                                        inp_0 = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample,
                                                        filter_dilation=filter_dilation, uniq_id=uniq_id)(inp[0])

                                        uniq_id += 1
                                        conv_fw = mkl_conv.Conv2D(imshp=imshp,
                                                                  kshp=kshp,
                                                                  border_mode=border_mode,
                                                                  subsample=subsample,
                                                                  filter_flip=filter_flip,
                                                                  filter_dilation=filter_dilation)(inp_0, weight, bias)

                                        uniq_id += 1
                                        gz = I2UGrad(uniq_id=uniq_id)(conv_fw, inp[1])

                                        uniq_id += 1
                                        out_0, out_1 = mkl_conv.ConvGradWeights(imshp=imshp,
                                                                                kshp=kshp,
                                                                                border_mode=border_mode,
                                                                                subsample=subsample,
                                                                                filter_flip=filter_flip,
                                                                                filter_dilation=filter_dilation,
                                                                                uniq_id=uniq_id)(image=inp_0, weight=weight, gradz=gz, bias=bias)
                                        # uniq_id += uniq_id
                                        # weightsGrad = U2IGrad(uniq_id=uniq_id)(weight, out_0)
                                        # uniq_id += uniq_id
                                        # biasGrad = U2IGrad(uniq_id=uniq_id)(bias, out_1)

                                        # Get BiasGrad
                                        oriBiasGrad = None  # BiasGrad in original function graph
                                        if isinstance(inp[1].owner.op, tensor.Elemwise):
                                            node_e = inp[1].owner
                                            if len(node_e.outputs[0].clients) >= 3:
                                                oriBiasGrad = self._check_grad_bias_(node_e, 0)
                                        elif isinstance(inp[1].owner.op, tensor.Split):
                                            node_e = inp[1].owner
                                            for i, split_out in enumerate(node_e.outputs):
                                                if inp[1] is split_out and len(split_out.clients) >= 3:
                                                    oriBiasGrad = self._check_grad_bias_(node_e, i)
                                        else:
                                            pass

                                        fgraph.replace_validate(out[0], out_0, 'ReplaceConvBias')
                                        if oriBiasGrad:
                                            fgraph.replace_validate(oriBiasGrad, out_1, 'ReplaceConvBias')
                                        did_something = True
                                    except Exception as e:
                                        raise
                elif (node in fgraph.apply_nodes) and isinstance(node.op, AbstractConv2d_gradInputs):
                    inp = node.inputs  # 0-weight, 1-gz, 2-shape
                    out = node.outputs
                    imshp = getattr(node.op, 'imshp', None)
                    kshp = getattr(node.op, 'kshp', None)
                    border_mode = getattr(node.op, 'border_mode', 'valid')
                    subsample = getattr(node.op, 'subsample', (1, 1))
                    filter_flip = getattr(node.op, 'filter_flip', False)
                    filter_dilation = getattr(node.op, 'filter_dilation', (1, 1))

                    assert len(inp) == 3 and len(out) == 1
                    list_Conv2D = [c[0] for c in inp[0].clients if (hasattr(c[0], 'op') and
                                                                    isinstance(c[0].op, mkl_conv.Conv2D) and
                                                                    len(c[0].inputs) == 3)]
                    if 3 > len(list_Conv2D) > 0:
                        if len(list_Conv2D) == 2:
                            assert list_Conv2D[0].op == list_Conv2D[1].op  # Can be merged in later optimization phase
                        x = list_Conv2D[0].inputs[0].owner.inputs[0]
                        bias = list_Conv2D[0].inputs[2]
                        inp_0 = list_Conv2D[0].inputs[0]
                        try:
                            uniq_id += 1
                            inp_0 = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample,
                                            filter_dilation=filter_dilation, uniq_id=uniq_id)(x)

                            uniq_id += 1
                            conv_fw = mkl_conv.Conv2D(imshp=imshp,
                                                      kshp=kshp,
                                                      border_mode=border_mode,
                                                      subsample=subsample,
                                                      filter_flip=filter_flip,
                                                      filter_dilation=filter_dilation)(inp_0, inp[0], bias)
                            uniq_id += 1
                            gz = I2UGrad(uniq_id=uniq_id)(conv_fw, inp[1])

                            uniq_id += 1
                            out_0 = mkl_conv.ConvGradInputs(imshp=imshp,
                                                            kshp=kshp,
                                                            border_mode=border_mode,
                                                            subsample=subsample,
                                                            filter_flip=filter_flip,
                                                            filter_dilation=filter_dilation)(inp_0, inp[0], gz)

                            uniq_id += 1
                            inp_grad = U2IGrad(uniq_id=uniq_id)(x, out_0)
                            fgraph.replace_validate(out[0], inp_grad, 'ReplaceConvBias')
                            # theano.printing.pydotprint(fgraph, outfile='replace_conv_input_grad.png', var_with_name_simple=True)
                            did_something = True
                        except Exception as e:
                            raise e
                else:
                    pass


# Register the instance of global OPT ReplaceConvBias into mkl_seqopt.
mkl_seqopt.register('MKL_CONV_REPLACE', ReplaceConvBias(), 0.095, 'fast_run', 'fast_compile', 'mkl')


# GLobal Optimizer for replace Elemwise_add with mkl_elemwise_sum
class ReplaceElemwise(Optimizer):
    def __init__(self):
        super(ReplaceElemwise, self).__init__()

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        # theano.printing.pydotprint(fgraph, outfile='xxxxxx.png', var_with_name_simple=True)
        global uniq_id

        def getElemwiseInput(node, inputs, coeffs, co):
            inp = inputs
            coe = coeffs

            # Elemwise_add
            if ((isinstance(node.op, tensor.Elemwise) and
                 isinstance(node.op.scalar_op, scalar.Add))):
                for i in node.inputs:
                    n = i.owner
                    if (n is not None and
                            isinstance(n.op, tensor.Elemwise) and
                            isinstance(n.op.scalar_op, scalar.Add)):
                        getElemwiseInput(n, inp, coe, co)
                    else:
                        inp.append(i)
                        coe.append(co)

            # Elemwise_mul: This case has been deleted.
            # We just process Elemwise{Add} here to avoid disturbing the Elemwise{Complesite} fusion.
            else:
                raise TypeError('The OP of the inputs node should be an instance of Elemwise{Add}')

        did_something = True
        while did_something:
            did_something = False
            topo = fgraph.toposort()
            topo.reverse()
            for node in topo:
                if node in fgraph.apply_nodes:
                    if (isinstance(node.op, tensor.Elemwise) and
                            isinstance(node.op.scalar_op, scalar.Add)):
                        out = node.outputs
                        inputs = []
                        coeffs = []
                        co = 1.0  # For now, all the coeffs are 1.0 since Elemwise{Mul} is not processed
                        getElemwiseInput(node, inputs, coeffs, co)
                        inp_len = len(inputs)
                        assert len(inputs) == len(coeffs)
                        if inp_len >= 2:
                            # print(inputs)
                            # print(coeffs)
                            # Check all inputs are from I2U and U2IGrad
                            if all([(i.owner and isinstance(i.owner.op, (I2U, U2IGrad))) for i in inputs]):
                                try:
                                    inputs_t = []
                                    for i in inputs:
                                        uniq_id += 1
                                        inputs_t.append(U2IElemwiseSum(inp_num=inp_len, coeff=coeffs, uniq_id=uniq_id)(i))
                                    uniq_id += 1
                                    out_t = mkl_elemwise.ElemwiseSum(inp_num=inp_len, coeff=coeffs, uniq_id=uniq_id)(*inputs_t)

                                    uniq_id += 1
                                    new_out = I2U(uniq_id=uniq_id)(out_t)
                                    fgraph.replace_validate(out[0], new_out, 'ReplaceElemwise')
                                    did_something = True
                                except Exception as e:
                                    raise e
                            else:
                                pass


# Register the instance of global OPT ReplaceElemwise into mkl_seqopt.
mkl_seqopt.register('MKL_ELEMWISE_REPLACE', ReplaceElemwise(), 30, 'fast_run', 'fast_compile', 'mkl')


@register_opt()
@local_optimizer([pool.Pool])
def local_pool_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, pool.Pool):
        return

    if node.inputs[0].type.ndim != 4:
        return

    if node.op.mode not in ('max', 'average_inc_pad', 'average_exc_pad'):
        return

    if not node.op.ignore_border:
        return

    x, ws, stride, pad = node.inputs
    if stride is None:
        stride = ws

    try:
        x_u2i = U2IPool(ignore_border=node.op.ignore_border,
                        mode=node.op.mode,
                        uniq_id=uniq_id)(x, ws, stride, pad)

        poolOut = mkl_pool.Pool(ignore_border=node.op.ignore_border,
                                mode=node.op.mode,
                                uniq_id=uniq_id)(x_u2i, ws, stride, pad)

        z_i2u = I2U(uniq_id=uniq_id)(poolOut)

        rval = z_i2u
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([pool.MaxPoolGrad, pool.AveragePoolGrad])
def local_poolGrad_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if node.inputs[0].type.ndim != 4:
        return

    if node.op.mode not in ('max', 'average_inc_pad', 'average_exc_pad'):
        return

    # currently, MKL only support this mode
    if not node.op.ignore_border:
        return

    if isinstance(node.op, pool.MaxPoolGrad):
        x, maxout, gz, ws, stride, pad = node.inputs
    elif isinstance(node.op, pool.AveragePoolGrad):
        x, gz, ws, stride, pad = node.inputs
    else:
        # Other pool mode is not supported
        return

    if stride is None:
        stride = ws

    try:
        x_u2i = U2IPool(ignore_border=node.op.ignore_border,
                        mode=node.op.mode,
                        uniq_id=uniq_id)(x, ws, stride, pad)

        poolOut = mkl_pool.Pool(ignore_border=node.op.ignore_border,
                                mode=node.op.mode,
                                uniq_id=uniq_id)(x_u2i, ws, stride, pad)

        gz_u2i = I2UGrad(uniq_id=uniq_id)(poolOut, gz)

        poolGradOut = mkl_pool.PoolGrad(ignore_border=node.op.ignore_border,
                                        mode=node.op.mode,
                                        uniq_id=uniq_id)(x_u2i, gz_u2i, ws, stride, pad)

        gx_i2u = U2IGrad(uniq_id=uniq_id)(x, poolGradOut)

        rval = gx_i2u
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([AbstractRelu])
def local_relu_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, AbstractRelu):
        return

    if node.inputs[0].type.ndim != 4:
        return

    x, = node.inputs

    try:
        x_u2i = U2IRelu(slope=node.op.slope, uniq_id=uniq_id)(x)
        reluOut = mkl_relu.Relu(slope=node.op.slope, uniq_id=uniq_id)(x_u2i)
        z_i2u = I2U(uniq_id=uniq_id)(reluOut)

        rval = z_i2u
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([AbstractReluGrad])
def local_reluGrad_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, AbstractReluGrad):
        return

    if node.inputs[0].type.ndim != 4:
        return

    x, gz = node.inputs

    try:
        x_u2i = U2IRelu(slope=node.op.slope, uniq_id=uniq_id)(x)
        reluOut = mkl_relu.Relu(slope=node.op.slope, uniq_id=uniq_id)(x_u2i)
        gz_u2i = I2UGrad(uniq_id=uniq_id)(reluOut, gz)

        reluGradOut = mkl_relu.ReluGrad(slope=node.op.slope, uniq_id=uniq_id)(x_u2i, gz_u2i)

        gx_i2u = U2IGrad(uniq_id=uniq_id)(x, reluGradOut)

        rval = gx_i2u
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([mkl_lrn.AbstractLRN])
def local_lrn_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, mkl_lrn.AbstractLRN):
        return

    x, = node.inputs
    x_u2i = U2ILRN(slope=node.op.slope,
                   uniq_id=uniq_id,
                   alpha=node.op.alpha,
                   beta=node.op.beta,
                   k=node.op.k,
                   n=node.op.n)(x)

    lrnout = mkl_lrn.LRN(uniq_id=uniq_id,
                         alpha=node.op.alpha,
                         beta=node.op.beta,
                         k=node.op.k,
                         n=node.op.n)(x_u2i)

    z_i2u = I2U(uniq_id=uniq_id)(lrnout)

    rval = z_i2u
    return [rval]


@register_opt()
@local_optimizer([mkl_lrn.AbstractLRNGrad])
def local_lrnGrad_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, mkl_lrn.AbstractLRNGrad):
        return

    x, gz, = node.inputs

    x_u2i = U2ILRN(slope=node.op.slope,
                   uniq_id=uniq_id,
                   alpha=node.op.alpha,
                   beta=node.op.beta,
                   k=node.op.k,
                   n=node.op.n)(x)

    lrnOut = mkl_lrn.LRN(uniq_id=uniq_id,
                         alpha=node.op.alpha,
                         beta=node.op.beta,
                         k=node.op.k,
                         n=node.op.n)(x_u2i)

    gz_u2i = I2UGrad(uniq_id=uniq_id)(lrnOut, gz)
    lrnGradOut = mkl_lrn.LRNGrad(uniq_id=uniq_id,
                                 alpha=node.op.alpha,
                                 beta=node.op.beta,
                                 k=node.op.k,
                                 n=node.op.n,
                                 fp=node.op.fp)(x_u2i, gz_u2i)

    gx_i2u = U2IGrad(uniq_id=uniq_id)(x, lrnGradOut)

    rval = gx_i2u
    return [rval]


@local_optimizer([AbstractConv2d])
def local_Conv2D_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d):
        return

    if node.op.filter_dilation != (1, 1):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    try:
        x, ws = node.inputs
        x_internal = U2IConv(imshp=node.op.imshp,
                             kshp=node.op.kshp,
                             subsample=node.op.subsample,
                             filter_dilation=node.op.filter_dilation,
                             uniq_id=uniq_id)(x)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation,
                                  uniq_id=uniq_id)(x_internal, ws)
        z_user = I2U(uniq_id=uniq_id)(convOut)
        reval = z_user
        return [reval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@local_optimizer([AbstractConv2d_gradInputs])
def local_ConvGradInputs_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    if node.op.filter_dilation != (1, 1):
        return

    try:
        ws, gz, topshp = node.inputs
        x = node.inputs[2].owner.inputs[0].owner.inputs[0]
        x_internal = U2IConv(imshp=node.op.imshp,
                             kshp=node.op.kshp,
                             subsample=node.op.subsample,
                             filter_dilation=node.op.filter_dilation,
                             uniq_id=uniq_id)(x)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation,
                                  uniq_id=uniq_id)(x_internal, ws)
        gz_internal = I2UGrad(uniq_id=uniq_id)(convOut, gz)
        dx = mkl_conv.ConvGradInputs(border_mode=node.op.border_mode,
                                     subsample=node.op.subsample,
                                     imshp=node.op.imshp,
                                     kshp=node.op.kshp)(x_internal, ws, gz_internal)
        dx_user = U2IGrad(uniq_id=uniq_id)(x, dx)
        rval = dx_user
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@local_optimizer([AbstractConv2d_gradWeights])
def local_ConvGradWeights_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    if node.op.filter_dilation != (1, 1):
        return

    try:
        x, gz, topshp = node.inputs
        ws = node.inputs[2].owner.inputs[0].owner.inputs[0]
        x_internal = U2IConv(imshp=node.op.imshp,
                             kshp=node.op.kshp,
                             subsample=node.op.subsample,
                             filter_dilation=node.op.filter_dilation,
                             uniq_id=uniq_id)(x)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation,
                                  uniq_id=uniq_id)(x_internal, ws)
        gz_internal = I2UGrad(uniq_id=uniq_id)(convOut, gz)
        dw = mkl_conv.ConvGradWeights(border_mode=node.op.border_mode,
                                      subsample=node.op.subsample,
                                      imshp=node.op.imshp,
                                      kshp=node.op.kshp)(x_internal, ws, gz_internal)
        rval = dw
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


conv_groupopt = theano.gof.optdb.LocalGroupDB()
conv_groupopt.__name__ = "mkl_conv_opts"
register_opt()(conv_groupopt)

# MKL-based convolution, using the same group with theano.tensor.nnet.opt to avoid dumlicating GEMM functions
# It can be disabled by excluding 'conv_mkl'.
conv_groupopt.register('local_Conv2D_mkl', local_Conv2D_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')
conv_groupopt.register('local_ConvGradInputs_mkl', local_ConvGradInputs_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')
conv_groupopt.register('local_ConvGradWeights_mkl', local_ConvGradWeights_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')


@register_opt()
@local_optimizer([mkl_bn.AbstractBatchNormalization])
def local_bn_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, mkl_bn.AbstractBatchNormalization):
        return

    x, scale, shift, = node.inputs[0:3]
    x_u2i = U2IBatchNormalization(eps=node.op.eps,
                                  uniq_id=uniq_id)(x)

    bn_out = mkl_bn.BatchNormalization(eps=node.op.eps,
                                       bias=node.op.bias,
                                       term=node.op.term,
                                       uniq_id=uniq_id)(x_u2i, scale, shift)

    z_i2u = I2U(uniq_id=uniq_id)(bn_out)
    rval = z_i2u
    return [rval]


@register_opt()
@local_optimizer([mkl_bn.AbstractBatchNormalizationGrad])
def local_bnGrad_mkl(node):
    global uniq_id
    uniq_id += 1

    if not mkl_available():
        return

    if not isinstance(node.op, mkl_bn.AbstractBatchNormalizationGrad):
        return

    x, gz, scale, shift, = node.inputs
    x_u2i = U2IBatchNormalization(eps=node.op.eps,
                                  uniq_id=uniq_id)(x)

    bn_out = mkl_bn.BatchNormalization(eps=node.op.eps,
                                       bias=node.op.bias,
                                       term=node.op.term,
                                       uniq_id=uniq_id)(x_u2i, scale, shift)

    gz_u2i = I2UGrad(uniq_id=uniq_id)(bn_out, gz)

    bn_GradOut = mkl_bn.BatchNormalizationGrad(eps=node.op.eps,
                                               bias=node.op.bias,
                                               term=node.op.term,
                                               uniq_id=uniq_id)(x_u2i, gz_u2i, scale, shift)

    gx_i2u = U2IGrad(uniq_id=uniq_id)(x, bn_GradOut[0])
    rval = [gx_i2u, bn_GradOut[1], bn_GradOut[2]]
    return rval

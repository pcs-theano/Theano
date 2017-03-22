
import theano
from theano import gof, tensor
from theano.sandbox.mkl import mkl_type
from theano.sandbox.mkl.mkl_type import MKLNdarrayType

from theano.sandbox.mkl.basic_ops import BaseConvertOp
from theano.tensor.nnet.abstract_conv import get_conv_output_shape

class I2U_Op(BaseConvertOp):
    __props__ = ()
    
    def c_support_code(self):
        ccode = """
            #define DIMENSION 4
            #define CHECK_ERR(f, err) \\
                    do { \\
                        (err) = (f); \\
                        if ((err) != E_SUCCESS) { \\
                            printf("Error in file [%s:%d], err code (%d)", \\
                                    __FILE__, __LINE__, err); \\
                            exit(1); \\
                        } \\
                    } while(0)
        """
        return ccode


    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "I2U"

    def make_node(self, x):
        if not isinstance(x.type, MKLNdarrayType):
            raise TypeError('Expected a Theano variable with type MKLNdarrayType.')
        return gof.Apply(self, [x], [tensor.TensorType(broadcastable=x.broadcastable, dtype=x.dtype)()])

    def c_code(self, node, name, inputs, outputs, sub):
        inp = inputs[0]
        out = outputs[0]
        fail = sub['fail']
        return """
        Py_XDECREF(%(out)s);
        %(out)s = (PyArrayObject*)MKLNdarray_CreateArrayObj(%(inp)s);
        if (!%(out)s) {
            %(fail)s;
        }
        """ % locals()

    def c_code_cache_version(self):
        return (1, 0, 0)


class U2I_BN(BaseConvertOp):
    __props__ = ('eps',)

    def __init__(self, eps=1e-5):
        self.eps = eps

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        return gof.Apply(self, [x], [MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()])

    def c_support_code(self):
        ccode = """
            #define DIMENSION 4
            #define CHECK_ERR(f, err) \\
                    do { \\
                        (err) = (f); \\
                        if ((err) != E_SUCCESS) { \\
                            printf("Error in file [%s:%d], err code (%d)", \\
                                    __FILE__, __LINE__, err); \\
                            exit(1); \\
                        } \\
                    } while(0)
        """
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        eps = self.eps
        fail = sub['fail']

        if 'float32' == node.inputs[0].type.dtype:
            typenum = 11
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            typenum = 12
            precision = 'F64'
        else:
            raise Exception('Type %s is not supported!' % node.inputs[0].type.dtype)

        ccode = """
        int ndim = PyArray_NDIM(%(x)s);
        int dtype = PyArray_TYPE(%(x)s);
        npy_intp* d = PyArray_DIMS(%(x)s);

        // assert (dtype == %(typenum)s);

        size_t dims[MAX_NDIM] = {0};
        for (int i = 0; i < ndim; i++) {
            dims[i] = (size_t)d[i];
        }

        Py_XDECREF(%(z)s);
        %(z)s = (MKLNdarray*)MKLNdarray_New(ndim, dtype);
        
        if (!%(z)s) {
            %(fail)s;
        }

        int status = MKLNdarray_set_structure(%(z)s, ndim, dims);
        if (status != 0) {
            %(fail)s;
        }

        bottomSize[0] = d[3];
        bottomSize[1] = d[2];
        bottomSize[2] = d[1];
        bottomSize[3] = d[0];

        bottomStride[0] = 1;
        bottomStride[1] = bottomStride[0] * bottomSize[0];
        bottomStride[2] = bottomStride[1] * bottomSize[1];
        bottomStride[3] = bottomStride[2] * bottomSize[2];

        CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, bottomSize, bottomStride), err);
        CHECK_ERR( dnnBatchNormalizationCreateForward_%(precision)s(&primitive, NULL, layout_user, %(eps)s), err);
        int ret = MKLNdarray_create_layout_buffer(%(z)s, &primitive, dnnResourceSrc);
        std::cout<<"ret:"<<ret<<std::endl; 
        if (!dnnLayoutCompare_%(precision)s(layout_user, %(z)s->private_layout)) {
            if (NULL == to_internal) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_user, %(z)s->private_layout), err);
            }
        }

        if (to_internal) {
            printf("bn convert\\n");
            CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal, PyArray_DATA(%(x)s), %(z)s->private_data), err);
        } else {
            printf("bn no convert \\n");
            memcpy(%(z)s->private_data, PyArray_DATA(%(x)s), dnnLayoutGetMemorySize_%(precision)s(%(z)s->private_layout));
        }

        // printf("BN CONVERT OK \\n");
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)



class I2IBN(BaseConvertOp):
    __props__ = ('eps',)

    def __init__(self, eps=1e-5):
        self.eps = eps


    def make_node(self, x):
        assert isinstance(x.type, MKLNdarrayType)

        return gof.Apply(self, [x], [x.type()])

    def c_support_code(self):
        ccode = """
            #define DIMENSION 4
            #define CHECK_ERR(f, err) \\
                    do { \\
                        (err) = (f); \\
                        if ((err) != E_SUCCESS) { \\
                            printf("Error in file [%s:%d], err code (%d)", \\
                                    __FILE__, __LINE__, err); \\
                            exit(1); \\
                        } \\
                    } while(0)
        """
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        eps = self.eps
        fail = sub['fail']

        if 'float32' == node.inputs[0].type.dtype:
            typenum = 11
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            typenum = 12
            precision = 'F64'
        else:
            raise Exception('Type %s is not supported' % x.type.dtype)

        ccode = """
            assert (MKLNdarray_Check((PyObject*)%(x)s));
            int ndim = %(x)s->nd;
            Py_XDECREF(%(z)s);
            %(z)s = (MKLNdarray*)MKLNdarray_New(%(x)s->nd, %(typenum)s);

            if (!%(z)s) {
                %(fail)s;
            }

            int status = MKLNdarray_set_structure(%(z)s, ndim, %(x)s->user_structure);
            if (status != 0) {
                %(fail)s;
            }

            CHECK_ERR( dnnBatchNormalizationCreateForward_%(precision)s(&primitive, NULL, %(x)s->private_layout, %(eps)s), err);
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&(%(z)s->private_layout), primitive, dnnResourceSrc), err);

            if (!dnnLayoutCompare_%(precision)s(%(x)s->private_layout, %(z)s->private_layout)) {
                if (NULL == to_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, %(x)s->private_layout, %(z)s->private_layout), err);
                }
            }

            CHECK_ERR( dnnAllocateBuffer_%(precision)s(&(%(z)s->private_data), %(z)s->private_layout), err);

            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal, %(x)s->private_data, %(z)s->private_data), err);
            } else {
                memcpy(%(z)s->private_data, %(x)s->private_data, dnnLayoutGetMemorySize_%(precision)s(%(z)s->private_layout));
            }

        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)


class U2I_Conv(BaseConvertOp):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1,1), filter_dilation=(1,1)):
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        self.imshp = imshp
        self.kshp = kshp
        self.filter_dilation = filter_dilation

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        return gof.Apply(self, [x], [MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()])

    def c_support_code(self):
        ccode = """
            #define DIMENSION 4
            #define CHECK_ERR(f, err) \\
                    do { \\
                        (err) = (f); \\
                        if ((err) != E_SUCCESS) { \\
                            printf("Error in file [%s:%d], err code (%d)", \\
                                    __FILE__, __LINE__, err); \\
                            exit(1); \\
                        } \\
                    } while(0)
        """
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        dH, dW = self.subsample

        if self.imshp is None:
            self.imshp = x.shape

        i_n, i_c, i_h, i_w = self.imshp

        if len(self.kshp) == 5:
            grp, k_n, k_c, k_h, k_w = self.kshp
        else:
            k_n, k_c, k_h, k_w = self.kshp
            grp = 1


	o_n, o_c, o_h, o_w = get_conv_output_shape(image_shape=self.imshp,
                                                   kernel_shape=self.kshp,
                                                   border_mode=self.border_mode,
                                                   filter_dilation=self.filter_dilation,
                                                   subsample=self.subsample)

        if self.border_mode == 'valid':
            padH, padW = (0, 0)
        elif self.border_mode == 'full':
            padH, padW = ((k_h - 1), (k_w - 1))
        elif self.border_mode == 'half':
            padH, padW = ((k_h / 2), (k_w / 2))
        elif isinstance(self.border_mode, tuple):
            padH, padW = self.border_mode
        else:
            raise ValueError("border_mode must have two elements")

        fail = sub['fail']

        if 'float32' == node.inputs[0].type.dtype:
            typenum = 11
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            typenum = 12
            precision = 'F64'
        else:
            raise Exception('Type %s is not supported!' % node.inputs[0].type.dtype)

        ccode = """
        int ndim = PyArray_NDIM(%(x)s);
        int dtype = PyArray_TYPE(%(x)s);
        npy_intp* d = PyArray_DIMS(%(x)s);

        // assert (dtype == %(typenum)s);

        size_t dims[32] = {0};

        int convPadding[2];
        size_t convStride[2], weightSize[5], weightStride[5], imageSize[4], imageStride[4], zSize[4], zStride[4];
        convStride[0] = %(dW)s;
        convStride[1] = %(dH)s;
        convPadding[0] = -%(padW)s;
        convPadding[1] = -%(padH)s;
        imageSize[0] = %(i_w)s;  //w
        imageSize[1] = %(i_h)s;  //h
        imageSize[2] = %(i_c)s;  //c
        imageSize[3] = %(i_n)s;  //n
        imageStride[0] = 1;
        imageStride[1] = imageSize[0];
        imageStride[2] = imageSize[0] * imageSize[1];
        imageStride[3] = imageSize[0] * imageSize[1] * imageSize[2];

        weightSize[0] = %(k_w)s;
        weightSize[1] = %(k_h)s;
        weightSize[2] = %(k_c)s;
        weightSize[3] = %(k_n)s;
        weightSize[4] = %(grp)s;
        weightStride[0] = 1;
        weightStride[1] = weightSize[0];
        weightStride[2] = weightSize[0] * weightSize[1];
        weightStride[3] = weightSize[0] * weightSize[1] * weightSize[2];
        weightStride[4] = weightSize[0] * weightSize[1] * weightSize[2] * weightSize[3];

        zSize[0] = %(o_w)s;
        zSize[1] = %(o_h)s;
        zSize[2] = %(o_c)s;
        zSize[3] = %(o_n)s;
        zStride[0] = 1;
        zStride[1] = zSize[0];
        zStride[2] = zSize[0] * zSize[1];
        zStride[3] = zSize[0] * zSize[1] * zSize[2];

        Py_XDECREF(%(z)s);
        %(z)s = (MKLNdarray*)MKLNdarray_New(ndim, dtype);
        
        if (!%(z)s) {
            %(fail)s;
        }
        
        for (int i = 0; i < ndim; i++) {
            dims[i] = (size_t)d[i];
        }

        int status = MKLNdarray_set_structure(%(z)s, ndim, dims);
        if (status != 0) {
            %(fail)s;
        }

        const int group = %(grp)s;
        //create user layout
        CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, imageSize, imageStride), err );
        CHECK_ERR( dnnConvolutionCreateForward_%(precision)s(&primitive, NULL,
                               dnnAlgorithmConvolutionDirect, DIMENSION, imageSize, zSize,
                               weightSize, convStride, convPadding, dnnBorderZeros), err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&(%(z)s->private_layout), primitive, dnnResourceSrc), err );

        if (!dnnLayoutCompare_%(precision)s(layout_user, %(z)s->private_layout)) {
            if (NULL == to_internal) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_user, %(z)s->private_layout), err );
            }
        }

        if (NULL == %(z)s->private_data) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&(%(z)s->private_data), %(z)s->private_layout), err );
        }

        if (to_internal) {
            printf("conv convert \\n");
            CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal, PyArray_DATA(%(x)s), %(z)s->private_data), err );
        } else {
            printf("conv no convert \\n");
            memcpy(%(z)s->private_data, PyArray_DATA(%(x)s), dnnLayoutGetMemorySize_%(precision)s(%(z)s->private_layout));
        }

        first_run = 0;

        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)



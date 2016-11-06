
"""
Ops for downsampling images.
Planned:
Pool, DownsampleAvg, DownsampleSoftmax.
"""
from __future__ import absolute_import, print_function, division
# This file should move along with conv.py
import warnings

import numpy
from six import integer_types
from six.moves import xrange
import six.moves.builtins as builtins

import theano
from theano.tensor.blas import ldflags
from theano import gof, OpenMPOp, tensor, Variable, Apply
from theano.tensor.nnet import mkldnn_helper


def max_pool_2d_same_size(input, patch_size):
    """
    Takes as input a 4-D tensor. It sets all non maximum values
    of non-overlapping patches of size (patch_size[0],patch_size[1]) to zero,
    keeping only the maximum values. The output has the same dimensions as
    the input.

    Parameters
    ----------
    input : 4-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    patch_size : tuple of length 2
        Size of the patch (patch height, patch width).
        (2,2) will retain only one non-zero value per patch of 4 values.

    """
    output = Pool(patch_size, True)(input)
    outs = MaxPoolGrad(patch_size, True)(input, output, output)
    return outs


def pool_2d(input, ds, ignore_border=None, st=None, padding=(0, 0),
            mode='max'):
    """Downscale the input by a specified factor

    Takes as input a N-D tensor, where N >= 2. It downscales the input image by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1])

    Parameters
    ----------
    input : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    ds : tuple of length 2
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    st : tuple of two ints
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding : tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.

    """
    if input.ndim < 2:
        raise NotImplementedError('pool_2d requires a dimension >= 2')
    if ignore_border is None:
        warnings.warn(
            "pool_2d() will have the parameter ignore_border"
            " default value changed to True (currently"
            " False). To have consistent behavior with all Theano"
            " version, explicitly add the parameter ignore_border=True."
            " On the GPU, using ignore_border=True is needed to use cuDNN."
            " When using ignore_border=False and not using cuDNN, the only"
            " GPU combination supported is when"
            " `ds == st and padding == (0, 0) and mode == 'max'`."
            " Otherwise, the convolution will be executed on CPU.",
            stacklevel=2)
        ignore_border = False
    if input.ndim == 4:
        op = Pool(ds, ignore_border, st=st, padding=padding,
                  mode=mode)
        output = op(input)
        return output

    # extract image dimensions
    img_shape = input.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = tensor.prod(input.shape[:-2])
    batch_size = tensor.shape_padright(batch_size, 1)

    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = tensor.cast(tensor.join(0, batch_size,
                                        tensor.as_tensor([1]),
                                        img_shape), 'int64')
    input_4D = tensor.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of images
    op = Pool(ds, ignore_border, st=st, padding=padding,
              mode=mode)
    output = op(input_4D)

    # restore to original shape
    outshp = tensor.join(0, input.shape[:-2], output.shape[-2:])
    return tensor.reshape(output, outshp, ndim=input.ndim)


class PoolBase(OpenMPOp):
    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0),
                 mode='max', openmp=None):
        super(PoolBase, self).__init__(openmp=openmp)  # remove openmp
        self.ds = tuple(ds)
        if not all([isinstance(d, integer_types) for d in ds]):
            raise ValueError(
                "Pool downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        assert isinstance(st, (tuple, list))
        self.st = tuple(st)
        self.ignore_border = ignore_border
        self.padding = tuple(padding)
        if self.padding != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
            raise NotImplementedError(
                'padding_h and padding_w must be smaller than strides')
        # if mode not in ['max', 'average_inc_pad', 'average_exc_pad', 'sum']:
        if mode not in ['max', 'min', 'average']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'min',"
                " 'average'. Got %s" % mode)
        self.mode = mode

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(PoolBase, self).c_compile_args()
        return compile_args

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_headers(self):
        headers = ['<stdio.h>', '<fstream>']
        headers += super(PoolBase, self).c_headers()
        return headers

    def c_support_code(self):
        return mkldnn_helper.mkldnn_header_text()

    def c_support_code_apply(self, node, name):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')
        
        sub = {}
        if dtype == 'float32':
            sub['dtype'] = 'float'
            sub['precision'] = 'F32'
        else:
            sub['dtype'] = 'double'
            sub['precision'] = 'F64'
        sub['name'] = name

        ccode = """
            #define __DEBUG__ 0
	    #define USER_LAYOUT 0
            #define dimension (4)
            #define CHECK_ERR(f, err) \\
                do { \\
                    (err) = (f); \\
                    if ((err) != E_SUCCESS) { \\
                        PyErr_Format(PyExc_RuntimeError, "Error in file " \\
                            "[%s:%d], err code (%d)", __FILE__, __LINE__, err); \\
                            exit(0); \\
                    } \\
                } while(0) 
        """

        ccode += """
            static int first_run = 1;
            static size_t inputSize[dimension] = {0};
            static size_t inputStrides[dimension] = {0};
            static size_t outputSize[dimension] = {0};
            static size_t outputStrides[dimension] = {0};
            static size_t kernelSize[2] = {0};
            static size_t kernelStride[2] = {0};
            static int inputOffset[2] = {0};

            static void *input_buffer_ptr = NULL;
            static void *input_buffer_ptr_from_previous = NULL;
            static void *input_buffer_ptr_to_previous = NULL;
            static void *output_buffer_ptr = NULL;
            static void *gz_buffer_ptr = NULL;
            static void *gz_buffer_tmp_ptr = NULL;
            static void *workspace_buffer_ptr = NULL;

            static dnnError_t err;
            static dnnPrimitiveAttributes_t attributes_%(name)s = NULL;
            static dnnPrimitive_t pPoolingFwd = NULL;
            static dnnPrimitive_t pPoolingBwd = NULL;
            static void *pool_res[dnnResourceNumber] = {0};
            static int input_buffer_size = 0;

            /////////////// only for debug usage ////////////////////
            size_t input_bytes;
            size_t output_bytes;
            size_t workspace_bytes;
            ////////////////////////////////////////////////////////

            ////FIXME, remove below definition if it's handled in conversion Op
            static dnnLayout_t usr_layout_input = NULL;
            static dnnLayout_t usr_layout_output = NULL;
            static dnnLayout_t int_layout_input = NULL;
            static dnnLayout_t *int_layout_input_ptr = NULL;
            static dnnLayout_t int_layout_input_from_previous = NULL;
            static dnnLayout_t int_layout_output = NULL;
            static dnnLayout_t gz_int_layout_from_other = NULL;
            static dnnLayout_t gz_int_layout = NULL;
            static dnnLayout_t int_layout_workspace = NULL;
            static dnnLayout_t *int_layout_workspace_p = NULL;
            static dnnPrimitive_t cvt_to_int_input = NULL;
            static dnnPrimitive_t cvt_gz_to_int = NULL;
            static dnnPrimitive_t cvt_from_int_input = NULL;
            static dnnPrimitive_t cvt_from_int_output = NULL;
            static dnnPrimitive_t convert_int2int_input = NULL;

            static void* bp[3];
            static unsigned int long ip;
            ////END
        """ % sub
        return ccode

    #def c_support_code_struct(self, node, name):
    #    ccode = """
    #    """
    #    return ccode

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"

        ccode = """
            //dnnDelete_%(precision)s(cvt_to_int_input);
            //dnnDelete_%(precision)s(cvt_gz_to_int);
            //dnnDelete_%(precision)s(cvt_from_int_input);
            //dnnDelete_%(precision)s(cvt_from_int_output);
            //dnnLayoutDelete_%(precision)s(usr_layout_input);
            //dnnLayoutDelete_%(precision)s(usr_layout_output);
            //dnnLayoutDelete_%(precision)s(int_layout_input);
            //dnnLayoutDelete_%(precision)s(int_layout_output);
            //dnnLayoutDelete_%(precision)s(int_layout_workspace);
        """ % locals()
        return ccode


class Pool(PoolBase):
    """
    For N-dimensional tensors, consider that the last two dimensions span
    images. This Op downsamples these images by taking the max, sum or average
    over different patch.

    The constructor takes the max, sum or average or different input patches.

    Parameters
    ----------
    ds : list or tuple of two ints
        Downsample factor over rows and column.
        ds indicates the pool region size.
    ignore_border : bool
        If ds doesn't divide imgshape, do we include an extra row/col
        of partial downsampling (False) or ignore it (True).
    st : list or tuple of two ints or None
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding: tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the images,
        pad_h is the size of the top and bottom margins, and pad_w is the size
        of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_inc_pad' excludes the padding from the count,
        'average_exc_pad' include it)

    """
    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode', 'uniq_id')

    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0),
                 mode='max', openmp=None, uniq_id=0):
        super(Pool, self).__init__(ds, ignore_border, st, padding, mode, openmp)
        self.uniq_id =  uniq_id
        self.fp = 'p_pooling'+str(uniq_id)

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        """
        Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple, list, or similar of integer or scalar Theano variable
            The shape of a tensor of images. The last two elements are
            interpreted as the number of rows, and the number of cols.
        ds : list or tuple of two ints
            Downsample factor over rows and columns this parameter indicates
            the size of the pooling region.
        st : list or tuple of two ints
            The stride size. This is the distance between the pooling regions.
            If it's set to None, it equals ds.
        ignore_border : bool
            If ds doesn't divide imgshape, do we include an extra row/col of
            partial downsampling (False) or ignore it (True).
        padding : tuple of two ints
            (pad_h, pad_w), pad zeros to extend beyond four borders
            of the images, pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins.

        Returns
        -------
        list
            The shape of the output from this op, for input of given shape.
            This will have the same length as imgshape, but with last two
            elements reduced as per the downsampling & ignore_border flags.

        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]
        r = tensor.extract_constant(r)
        c = tensor.extract_constant(c)
        
        # use Intel-Caffe style calculation
        out_r = numpy.ceil((r + 2 * padding[0] - ds[0]) / float(st[0])) + 1
        out_c = numpy.ceil((c + 2 * padding[1] - ds[1]) / float(st[1])) + 1
        
        if (padding[0] or padding[1]):
            if ((out_r - 1) * st[0]) >= (r + padding[0]):
                out_r -= 1
            if ((out_c - 1) * st[1]) >= (c + padding[1]):
                out_c -= 1
            assert(((out_r - 1) * st[0]) < (r + padding[0]))
            assert(((out_c - 1) * st[1]) < (c + padding[1]))

        if isinstance(out_r, theano.Variable):
            nr = tensor.cast(out_r, 'int32')
        else:
            nr = numpy.int(out_r)

        if isinstance(out_c, theano.Variable):
            nc = tensor.cast(out_c, 'int32')
        else:
            nc = numpy.int(out_c)

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def make_node(self, x):
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError()
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return gof.Apply(self, [x], [out()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,
                                 self.padding)
        if not self.ignore_border:
            assert z_shape[2] > 0
            assert z_shape[3] > 0
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = numpy.empty(z_shape, dtype=x.dtype)
        zz = z[0]
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w
        inc_pad = self.mode == 'average_inc_pad'

        # pad the image
        if self.padding != (0, 0):
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x
        func = numpy.max
        if self.mode == 'sum':
            func = numpy.sum
        elif self.mode != 'max':
            func = numpy.average

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = builtins.min(row_st + ds0, img_rows)
                    if not inc_pad:
                        row_st = builtins.max(row_st, self.padding[0])
                        row_end = builtins.min(row_end, x.shape[-2] + pad_h)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = builtins.min(col_st + ds1, img_cols)
                        if not inc_pad:
                            col_st = builtins.max(col_st, self.padding[1])
                            col_end = builtins.min(col_end,
                                                   x.shape[-1] + pad_w)
                        zz[n, k, r, c] = func(y[
                            n, k, row_st:row_end, col_st:col_end])

    '''
    # infer_shape will cause shape info crash during opt when running googlenet v1
    def infer_shape(self, node, in_shapes):
        shp = self.out_shape(in_shapes[0], self.ds,
                             self.ignore_border, self.st, self.padding)
        return [shp]
    '''

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [PoolGrad(self.ds,
                         ignore_border=self.ignore_border,
                         st=self.st,
                         padding=self.padding,
                         uniq_id=self.uniq_id,
                         mode=self.mode,
                         fp=self.fp)(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        bottom, = inp
        top, = out
        fp=self.fp
        
        dH, dW = self.ds
        sH, sW = self.st
        padH, padW = self.padding

        if self.mode == 'max':
            mode = 'Max'
        elif self.mode == 'min':
            mode = 'Min'
        elif self.mode == 'average':
            mode = 'Avg'
        else:
            raise VauleError("mode must be one of 'max', 'min', 'average'")

        ignore_border = int(self.ignore_border)
        #if self.ignore_border:
        #    borderType = 'dnnBorderZeros'
        #else:
        #    borderType = 'dnnBorderExtrapolation'
        # current mkl only support this type
        borderType = 'dnnBorderZeros'

        if node.inputs[0].type.dtype == "float32":  # FIXME, remove if it's defined in other place
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'
        else:
            raise TypeError('input must be float32 or float64')

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        #if __DEBUG__
        std::cout<<"pool start"<<std::endl;
        #endif
        if (1 == first_run){
            FILE *pFile;
            pFile = fopen("%(fp)s","w");
            fprintf(pFile,"%%llu", bp);
            fflush(pFile);
            fclose(pFile);
        }
        if (1 == first_run) {
            size_t kernel_h = %(dH)s;
            size_t kernel_w = %(dW)s;
            size_t stride_h = %(sH)s;
            size_t stride_w = %(sW)s;
            size_t pad_h = %(padH)s;
            size_t pad_w = %(padW)s;
    
            kernelSize[0] = kernel_w;
            kernelSize[1] = kernel_h;
            kernelStride[0] = stride_w;
            kernelStride[1] = stride_h;
            inputOffset[0] = -pad_w;
            inputOffset[1] = -pad_h;
    
            int out_h, out_w; // shape of the output
            int in_h, in_w; // shape of the padded_input
            in_h = PyArray_DIMS(%(bottom)s)[2];
            in_w = PyArray_DIMS(%(bottom)s)[3];
    
            // using Intel-Caffe style to calculate the output shape
            out_h = ceil((float)(in_h + 2 * pad_h - kernel_h)/stride_h) + 1;
            out_w = ceil((float)(in_w + 2 * pad_w - kernel_w)/stride_w) + 1;
            if (pad_h || pad_w) {
                if ((out_h - 1) * stride_h >= (in_h + pad_h)) {
                    --out_h;
                }
                if ((out_w - 1) * stride_w >= (in_w + pad_w)) {
                    --out_w;
                }
                assert((out_h - 1) * stride_h < in_h + pad_h);
                assert((out_w - 1) * stride_w < in_w + pad_w);
            }
    
            inputSize[0] = PyArray_DIMS(%(bottom)s)[3];  //w
            inputSize[1] = PyArray_DIMS(%(bottom)s)[2];  //h
            inputSize[2] = PyArray_DIMS(%(bottom)s)[1];  //c
            inputSize[3] = PyArray_DIMS(%(bottom)s)[0];  //n
            inputStrides[0] = 1;
            inputStrides[1] = inputSize[0];
            inputStrides[2] = inputSize[0] * inputSize[1];
            inputStrides[3] = inputSize[0] * inputSize[1] * inputSize[2];
    
            outputSize[0] = out_w;
            outputSize[1] = out_h;
            outputSize[2] = inputSize[2];
            outputSize[3] = inputSize[3];
            outputStrides[0] = 1;
            outputStrides[1] = outputSize[0];
            outputStrides[2] = outputSize[0] * outputSize[1];
            outputStrides[3] = outputSize[0] * outputSize[1] * outputSize[2];
    
            CHECK_ERR( dnnPrimitiveAttributesCreate_%(precision)s(&attributes_%(name)s), err );
        }
        #if __DEBUG__
            std::cout << "inputSize: " << inputSize[3] << "x" << inputSize[2] << "x" << inputSize[1] << "x" << inputSize[0] << std::endl;
            std::cout << "outputSize: " << outputSize[3] << "x" << outputSize[2] << "x" << outputSize[1] << "x" << outputSize[0] << std::endl;
            std::cout << "pooling region: " << kernelSize[0] << "x" << kernelSize[1] << std::endl;
            std::cout << "pooling stride: " << kernelStride[0] << "x" << kernelStride[1] << std::endl;
            std::cout << "padding: " << inputOffset[0] << "x" << inputOffset[1] << std::endl;
        #endif
        
    
        // get internal layout for topgrad from previous Op
        int_layout_input_from_previous = ((dnnLayout_t*)PyArray_DATA(%(bottom)s))[0];
        // get internal buffer for topgrad from previous op
        input_buffer_ptr_from_previous = ((void **)PyArray_DATA(%(bottom)s))[1];

        #if __DEBUG__
	    std::cout <<"++++++++++++\\pool forward, int_layout_input_from_previous: @"<<int_layout_input_from_previous<<std::endl;
	    std::cout <<"pool forward, input_buffer_ptr_from_previous: @"<<input_buffer_ptr_from_previous<<std::endl;
        #endif
    
        if (NULL == pPoolingFwd) {
            CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&pPoolingFwd, NULL,
                       dnnAlgorithmPooling%(mode)s, int_layout_input_from_previous, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }
    
        if (NULL == int_layout_input) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &int_layout_input, pPoolingFwd, dnnResourceSrc), err );
        }
        if (NULL == int_layout_output) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &int_layout_output, pPoolingFwd, dnnResourceDst), err );
        }
        if (NULL == int_layout_workspace) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &int_layout_workspace, pPoolingFwd, dnnResourceWorkspace), err );
        }

        if (NULL == output_buffer_ptr) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&output_buffer_ptr, int_layout_output) , err );
        }
        if (NULL == workspace_buffer_ptr) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&workspace_buffer_ptr, int_layout_workspace) , err );
        }
        pool_res[dnnResourceWorkspace] = workspace_buffer_ptr;
        ((dnnLayout_t**)bp)[0] = &int_layout_workspace;
        ((void**)bp)[1] = workspace_buffer_ptr;

        npy_intp out_dim[4];
        out_dim[0] = outputSize[3];
        out_dim[1] = outputSize[2];
        out_dim[2] = outputSize[1];
        out_dim[3] = outputSize[0];
        // Prepare output array
        int typenum;
        //if ( !(%(top)s
        //        && PyArray_NDIM(%(top)s) == 4
        //        && PyArray_IS_C_CONTIGUOUS(%(top)s)
        //        && PyArray_DIMS(%(top)s)[0] == out_dim[0]
        //        && PyArray_DIMS(%(top)s)[1] == out_dim[1]
        //        && PyArray_DIMS(%(top)s)[2] == out_dim[2]
        //        && PyArray_DIMS(%(top)s)[3] == out_dim[3])) {
        //    Py_XDECREF(%(top)s);
        if ( !(%(top)s) ) {
            typenum = PyArray_TYPE(%(bottom)s);
            %(top)s = (PyArrayObject*)PyArray_ZEROS(dimension,
                                              out_dim,
                                              typenum,
                                              0);
            if (NULL == %(top)s) {
                PyErr_Format(PyExc_RuntimeError,
                            "PoolBase: Failed to allocate output of %%lld x %%lld x %%lld x %%lld",
                            (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3]);
                %(fail)s
            }
        }

        if (!dnnLayoutCompare_%(precision)s(int_layout_input_from_previous, int_layout_input)) {
            #if 1 || __DEBUG__
                std::cout<<"############ pool forward, input layout is not equal" <<std::endl;
            #endif                                                           
            if (NULL == convert_int2int_input) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_int2int_input, int_layout_input_from_previous, int_layout_input), err );
            }
        }
        if (convert_int2int_input) {
            if (NULL == input_buffer_ptr) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&input_buffer_ptr, int_layout_input), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_input, input_buffer_ptr_from_previous, input_buffer_ptr), err );
            int_layout_input_ptr = &int_layout_input;
        } else {                                                             
            int_layout_input_ptr = &int_layout_input_from_previous;          
            input_buffer_ptr = input_buffer_ptr_from_previous;
        }

        pool_res[dnnResourceSrc] = input_buffer_ptr;
        pool_res[dnnResourceDst] = output_buffer_ptr;

        #if __DEBUG__
        input_bytes = dnnLayoutGetMemorySize_%(precision)s(*int_layout_input_ptr);
        output_bytes = dnnLayoutGetMemorySize_%(precision)s(int_layout_output);
        workspace_bytes = dnnLayoutGetMemorySize_%(precision)s(int_layout_workspace);
        std::cout << " input_bytes = " << input_bytes << std::endl;
        std::cout << " output_bytes = " << output_bytes << std::endl;
        std::cout << " workspace_bytes =  " << workspace_bytes << std::endl;
        std::cout << "pool_res[dnnResourceSrc] = @" << pool_res[dnnResourceSrc] << std::endl;
        std::cout << "pool_res[dnnResourceDst] = @" << pool_res[dnnResourceDst] << std::endl;
        std::cout << "pool_res[dnnResourceWorkspace] = @" << pool_res[dnnResourceWorkspace] << std::endl;
        #endif

        CHECK_ERR( dnnExecute_%(precision)s(pPoolingFwd, (void**)pool_res), err );

        ((dnnLayout_t*)PyArray_DATA(%(top)s))[0] = int_layout_output;
        ((void**)PyArray_DATA(%(top)s))[1] = output_buffer_ptr;

        #if 0
            float *out_p = (float *)workspace_buffer_ptr;
            printf(\"pool forward, workspace; %%g, %%g, %%g, %%g, %%g\\n\", out_p[0], out_p[1],out_p[2],out_p[3],out_p[4]);
            if (dnnLayoutGetMemorySize_%(precision)s(int_layout_output) != (outputSize[0] * outputSize[1] * outputSize[2] * outputSize[3] * sizeof(%(dtype)s))) {
                printf(\"ERROR: conv forward, z view size NOT equal with z_layout!!!!!!\\n\");
            }
        #endif

        first_run = 0;
        #if __DEBUG__
        std::cout<<"pool forward, output_buffer_ptr: @"<<output_buffer_ptr<<", output layout: @"<<int_layout_output<<std::endl;
        std::cout<<"pool end\\n"<<std::endl;
        #endif
        """ % sub

        return ccode

    def c_code_cache_version(self):
        return (0, 6, 8, 4, self.openmp, self.uniq_id)


class PoolGrad(PoolBase):
    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode', 'uniq_id')

    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0),
                 mode='max', openmp=None, uniq_id=0, fp="default.txt"):
        super(PoolGrad, self).__init__(ds, ignore_border, st, padding, mode, openmp)
        self.uniq_id = uniq_id
        self.fp = fp

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        """Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple of integers or scalar Theano variables
            the shape of a tensor of images. The last two elements are
            interpreted as the number of rows, and the number of cols.
        ds : tuple of two ints
            downsample factor over rows and columns this parameter
            indicates the size of the pooling region
        st : tuple of two ints
            the stride size. This is the distance between the pooling
            regions. If it's set to None, in which case it equlas ds.
        ignore_border : bool
            if ds doesn't divide imgshape, do we include an extra
            row/col of partial downsampling (False) or ignore it
            (True).
        padding : tuple of two ints
            (pad_h, pad_w), pad zeros to extend beyond four borders of
            the images, pad_h is the size of the top and bottom
            margins, and pad_w is the size of the left and right
            margins.

        Returns
        -------
        list :
            the shape of the output from this op, for input of given
            shape.  This will have the same length as imgshape, but
            with last two elements reduced as per the downsampling &
            ignore_border flags.

        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]

        # use Intel-Caffe style calculation
        out_r = numpy.ceil((r + 2 * padding[0] - ds[0]) / float(st[0])) + 1
        out_c = numpy.ceil((c + 2 * padding[1] - ds[1]) / float(st[1])) + 1
        
        if (padding[0] or padding[1]):
            if ((out_r - 1) * st[0]) >= (r + padding[0]):
                out_r -= 1
            if ((out_c - 1) * st[1]) >= (c + padding[1]):
                out_c -= 1
            assert(((out_r - 1) * st[0]) < (r + padding[0]))
            assert(((out_c - 1) * st[1]) < (c + padding[1]))

        if isinstance(out_r, theano.Variable):
            nr = tensor.cast(out_r, 'int32')
        else:
            nr = numpy.int(out_r)

        if isinstance(out_c, theano.Variable):
            nc = tensor.cast(out_c, 'int32')
        else:
            nc = numpy.int(out_c)

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    '''
    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]
    '''

    def make_node(self, x, gz):
        x = tensor.as_tensor_variable(x)
        gz = tensor.as_tensor_variable(gz)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4

        return Apply(self, [x, gz], [x.type()])

    def c_code(self, node, name, inp, out, sub):
        bottom, topgrad = inp
        bottomgrad, = out

        dH, dW = self.ds
        sH, sW = self.st
        padH, padW = self.padding

        if self.mode == 'max':
            mode = 'Max'
        elif self.mode == 'min':
            mode = 'Min'
        elif self.mode == 'average':
            mode = 'Avg'
        else:
            raise VauleError("mode must be one of 'max', 'min', 'average'")

        fp = self.fp

        ignore_border = int(self.ignore_border)
        #if self.ignore_border:
        #    borderType = 'dnnBorderZeros'
        #else:
        #    borderType = 'dnnBorderExtrapolation'
        borderType = 'dnnBorderZeros'

        if node.inputs[0].type.dtype == "float32":  # FIXME, remove if it's defined in other place
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        #if __DEBUG__
        std::cout<<"poolgrad start"<<std::endl;
        #endif
        if(first_run) {
            std::ifstream inp("%(fp)s");
            if(inp.is_open()) {
                std::cout<<"pool open inp ok\\n";
            }else{
                std::cout<<"open fail\\n";
            }
            while(inp>>ip){
                //std::cout<<std::hex<<"ip "<<ip<<std::dec<<std::endl;
            }
            inp.close();
        }
        if (1 == first_run) {
            size_t kernel_h = %(dH)s;
            size_t kernel_w = %(dW)s;
            size_t stride_h = %(sH)s;
            size_t stride_w = %(sW)s;
            size_t pad_h = %(padH)s;
            size_t pad_w = %(padW)s;
    
            kernelSize[0] = kernel_w;
            kernelSize[1] = kernel_h;
            kernelStride[0] = stride_w;
            kernelStride[1] = stride_h;
            inputOffset[0] = -pad_w;
            inputOffset[1] = -pad_h;
    
            ///////////// no need to calc output h&w since we can get the shape from topgrad. remove it.
            //int out_h, out_w; // shape of the output
            //int in_h, in_w; // shape of the padded_input
            //in_h = PyArray_DIMS(%(bottom)s)[2];
            //in_w = PyArray_DIMS(%(bottom)s)[3];
    
            //// using Intel-Caffe style to calculate the output shape
            //out_h = ceil((float)(in_h + 2 * pad_h - kernel_h)/stride_h) + 1;
            //out_w = ceil((float)(in_w + 2 * pad_w - kernel_w)/stride_w) + 1;
            //if (pad_h || pad_w) {
            //    if ((out_h - 1) * stride_h >= (in_h + pad_h)) {
            //        --out_h;
            //    }
            //    if ((out_w - 1) * stride_w >= (in_w + pad_w)) {
            //        --out_w;
            //    }
            //    assert((out_h - 1) * stride_h < in_h + pad_h);
            //    assert((out_w - 1) * stride_w < in_w + pad_w);
            //}
            ///////////////////////////////////////////////////////////////////////////////
    
            //use 'bottom' instead of '%(bottom)s' will cause segment fault!!!
            inputSize[0] = PyArray_DIMS(%(bottom)s)[3];  //w
            inputSize[1] = PyArray_DIMS(%(bottom)s)[2];  //h
            inputSize[2] = PyArray_DIMS(%(bottom)s)[1];  //c
            inputSize[3] = PyArray_DIMS(%(bottom)s)[0];  //n
            inputStrides[0] = 1;
            inputStrides[1] = inputSize[0];
            inputStrides[2] = inputSize[0] * inputSize[1];
            inputStrides[3] = inputSize[0] * inputSize[1] * inputSize[2];
    
            outputSize[0] = PyArray_DIMS(%(topgrad)s)[3];
            outputSize[1] = PyArray_DIMS(%(topgrad)s)[2];
            outputSize[2] = PyArray_DIMS(%(topgrad)s)[1];
            outputSize[3] = PyArray_DIMS(%(topgrad)s)[0];
            outputStrides[0] = 1;
            outputStrides[1] = outputSize[0];
            outputStrides[2] = outputSize[0] * outputSize[1];
            outputStrides[3] = outputSize[0] * outputSize[1] * outputSize[2];
    
            #if __DEBUG__
            std::cout << "inputgradSize: " << inputSize[3] << "x" << inputSize[2] << "x" << inputSize[1] << "x" << inputSize[0] << std::endl;
            std::cout << "outputgradSize: " << outputSize[3] << "x" << outputSize[2] << "x" << outputSize[1] << "x" << outputSize[0] << std::endl;
            std::cout << "pooling region: " << kernelSize[0] << "x" << kernelSize[1] << std::endl;
            std::cout << "pooling stride: " << kernelStride[0] << "x" << kernelStride[1] << std::endl;
            std::cout << "padding: " << inputOffset[0] << "x" << inputOffset[1] << std::endl;
            #endif
    
            CHECK_ERR( dnnPrimitiveAttributesCreate_%(precision)s(&attributes_%(name)s), err );
        }
    
        // get internal layout for topgrad from previous Op
        int_layout_input_from_previous = ((dnnLayout_t*)PyArray_DATA(%(bottom)s))[0];
    
        if (NULL == pPoolingBwd) {
            CHECK_ERR( dnnPoolingCreateBackward_%(precision)s(&pPoolingBwd, NULL,
                       dnnAlgorithmPooling%(mode)s, int_layout_input_from_previous, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }

        if (NULL == pPoolingFwd) {
            CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&pPoolingFwd, NULL,
                       dnnAlgorithmPooling%(mode)s, int_layout_input_from_previous, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }

        if (NULL == int_layout_input) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &int_layout_input, pPoolingFwd, dnnResourceSrc), err );
        }
        if (NULL == gz_int_layout) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &gz_int_layout, pPoolingFwd, dnnResourceDst), err );
        }

        if (NULL == input_buffer_ptr) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&input_buffer_ptr, int_layout_input) , err );
            input_buffer_size = dnnLayoutGetMemorySize_%(precision)s(int_layout_input);
        }
        #pragma omp parallel for
        #pragma ivdep
        for(int i = 0 ; i < input_buffer_size/sizeof(%(dtype)s); ++i) {
             ((unsigned int*)input_buffer_ptr)[i] = 0;
        }
        //memset(input_buffer_ptr, 0, dnnLayoutGetMemorySize_%(precision)s(int_layout_input));
        // Prepare output array
        int typenum;
        if (!(%(bottomgrad)s)) {

            typenum = PyArray_TYPE(%(topgrad)s);
            %(bottomgrad)s = (PyArrayObject*)PyArray_ZEROS(4,
                                                  PyArray_DIMS(%(bottom)s),
                                                  typenum,
                                                  0);
            if (NULL == %(bottomgrad)s) {
                std::cout<<"alocat fail\\n";
            }
        }
   
        // get internal buffer for topgrad from previous op
        gz_int_layout_from_other = ((dnnLayout_t*)PyArray_DATA(%(topgrad)s))[0];
        gz_buffer_ptr = ((void **)PyArray_DATA(%(topgrad)s))[1];
      
        int_layout_workspace_p = ((dnnLayout_t**)ip)[0];
        int_layout_workspace = *int_layout_workspace_p;
        pool_res[dnnResourceWorkspace] = ((void**)ip)[1];

        if(first_run ==1)
        { 
            if (!dnnLayoutCompare_%(precision)s(gz_int_layout_from_other, gz_int_layout)) {
            #if __DEBUG__
                std::cout<<"############ pool backward, gz layout is not equal" <<std::endl;
            #endif                                                           
                if (NULL == cvt_gz_to_int) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&cvt_gz_to_int, gz_int_layout_from_other, gz_int_layout), err );
                 }
            }
        }

        if (cvt_gz_to_int) {
            if (NULL == gz_buffer_tmp_ptr) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gz_buffer_tmp_ptr, gz_int_layout), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(cvt_gz_to_int, gz_buffer_ptr, gz_buffer_tmp_ptr), err );
        } else {
             gz_buffer_tmp_ptr = gz_buffer_ptr;
        }
        pool_res[dnnResourceDiffDst] = gz_buffer_tmp_ptr;
        pool_res[dnnResourceDiffSrc] = input_buffer_ptr;

        #if __DEBUG__
        input_bytes = dnnLayoutGetMemorySize_%(precision)s(int_layout_input);
        output_bytes = dnnLayoutGetMemorySize_%(precision)s(gz_int_layout);
        workspace_bytes = dnnLayoutGetMemorySize_%(precision)s(int_layout_workspace);
        std::cout << " input_bytes = " << input_bytes << std::endl;
        std::cout << " output_bytes = " << output_bytes << std::endl;
        std::cout << " workspace_bytes =  " << workspace_bytes << std::endl;
        #endif

        CHECK_ERR( dnnExecute_%(precision)s(pPoolingBwd, (void**)pool_res), err );

        if (!dnnLayoutCompare_%(precision)s(int_layout_input, int_layout_input_from_previous)) {
            #if __DEBUG__
                std::cout<<"############ pool backward, input layout is not equal" <<std::endl;
            #endif                                                           
            if (NULL == convert_int2int_input) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_int2int_input, int_layout_input, int_layout_input_from_previous), err );
            }
        } 
        if (convert_int2int_input) {
            if (NULL == input_buffer_ptr_to_previous) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&input_buffer_ptr_to_previous, int_layout_input_from_previous ), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_input, input_buffer_ptr, input_buffer_ptr_to_previous), err );
         } else {                                                          
            input_buffer_ptr_to_previous = input_buffer_ptr;
            //printf(\"D2: %%x\\n\",((dnnLayout_t*)PyArray_DATA(%(bottomgrad)s))[0]);
        } 


        ((dnnLayout_t*)PyArray_DATA(%(bottomgrad)s))[0] = int_layout_input_from_previous;
        ((void**)PyArray_DATA(%(bottomgrad)s))[1] = input_buffer_ptr_to_previous;

        first_run = 0;
        
        """ % sub

        return ccode

    def c_code_cache_version(self):
        return (0, 6, 8, 4, self.openmp, self.uniq_id)


class MaxPoolGrad(PoolGrad):
    def __init__(self, ds, ignore_border, st=None, padding=(0, 0), openmp=None):
        PoolGrad.__init__(self, ds, ignore_border, st, padding, 'max', openmp)

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(maxout, Variable) and maxout.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, inp, out):
        assert self.mode == 'max'
        x, maxout, gz = inp
        gx_stg, = out
        # number of pooling output rows
        pr = maxout.shape[-2]
        # number of pooling output cols
        pc = maxout.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w

        # pad the image
        if self.padding != (0, 0):
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x
        gx = numpy.zeros_like(y)
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = builtins.max(r * st0, self.padding[0])
                    row_end = builtins.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = builtins.max(c * st1, self.padding[1])
                        col_end = builtins.min(col_st + ds1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == y[n, k, row_ind, col_ind]):
                                    gx[n, k, row_ind, col_ind] += gz[n, k, r, c]
        # unpad the image
        gx = gx[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)]
        gx_stg[0] = gx

    def grad(self, inp, grads):
        x, maxout, gz = inp
        ggx, = grads
        return [theano.tensor.zeros_like(x),
                theano.tensor.zeros_like(maxout),
                DownsampleFactorMaxGradGrad(
                    self.ds, ignore_border=self.ignore_border,
                    st=self.st, padding=self.padding)(x, maxout, ggx)]

    def c_code(self, node, name, inp, out, sub):
        assert self.mode == 'max'
        x, z, gz = inp
        gx, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, maximum) schedule(static)'
        else:
            omp_parallel = ''
        return """
        // sanity checks
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_typenum = PyArray_ObjectType((PyObject*)%(z)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
        if ((x_typenum != z_typenum) || (x_typenum != gz_typenum))
        {
            PyErr_SetString(PyExc_ValueError, "input types must all match");
            %(fail)s;
        }
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(z)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "z must be a 4d ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(gz)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "gz must be a 4d ndarray");
            %(fail)s;
        }
        int z_r, z_c;
        z_r = PyArray_DIMS(%(z)s)[2];
        z_c = PyArray_DIMS(%(z)s)[3];
        int r, c; // shape of the padded_input
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += %(pd0)s * 2;
        c += %(pd1)s * 2;
        // allocating memory for gx
        if ((!%(gx)s)
          || !PyArray_ISCONTIGUOUS(%(gx)s)
          || *PyArray_DIMS(%(gx)s)!=4
          ||(PyArray_DIMS(%(gx)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(gx)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(gx)s)[2] != PyArray_DIMS(%(x)s)[2])
          ||(PyArray_DIMS(%(gx)s)[3] != PyArray_DIMS(%(x)s)[3])
          )
        {
          Py_XDECREF(%(gx)s);
          %(gx)s = (PyArrayObject*) PyArray_ZEROS(4, PyArray_DIMS(%(x)s), x_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(gx)s, 0);
        }
        dtype_%(z)s maximum; // temp var for maximum value in a region
        if (z_r && z_c)
        {
            int r_st, r_end, c_st, c_end;
            %(omp_parallel)s
            for(int t = 0; t < PyArray_DIMS(%(x)s)[0] * PyArray_DIMS(%(x)s)[1]; t++){
                int b = t %% PyArray_DIMS(%(x)s)[0];
                int k = t / PyArray_DIMS(%(x)s)[0];
                for(int i=0; i < z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    // change coordinates from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // the maximum value
                    maximum = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,i,j)))[0];
                    // the gradient corresponding to this maximum value in z
                    dtype_%(gz)s * gz = (
                          (dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s, b, k, i, j)));
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        dtype_%(gx)s * gx = (
                          (dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s, b, k, m, n)));
                        if (a == maximum){
                          gx[0] = gx[0] + gz[0];
                        }
                      }
                    }
                  }
                }
              }
            }
            printf(\"normMaxPool z:%%x\\n\",%(gx)s);
        """ % locals()

    def c_code_cache_version(self):
        return (0, 7, self.openmp)


class AveragePoolGrad(PoolGrad):
    def __init__(self, ds, ignore_border, st=None, padding=(0, 0),
                 mode='average_inc_pad'):
        assert mode in ['sum', 'average_inc_pad', 'average_exc_pad']
        PoolGrad.__init__(self, ds, ignore_border, st, padding, mode)

    # There is an extra dummy parameter to match the parameter count
    # of MaxPoolGrad.  They have to keep the same interface because of
    # the DownsampleFactorMaxGrad trick to keep old scripts working
    # (see downsample.py for details on this).
    def make_node(self, x, gz, dummy=None):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        gz = tensor.as_tensor_variable(gz)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4

        return Apply(self, [x, gz], [x.type()])

    def perform(self, node, inp, out):
        if self.mode == 'average_exc_pad' and self.padding != (0, 0):
            raise NotImplementedError()
        x, gz = inp
        gx_stg, = out
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,
                                 self.padding)
        if (gx_stg[0] is None) or (gx_stg[0].shape != z_shape):
            gx_stg[0] = numpy.empty(z_shape, dtype=x.dtype)
        zz = gx_stg[0]
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w
        inc_pad = self.mode == 'average_inc_pad'
        sum_mode = self.mode == 'sum'

        # pad the image
        if self.padding != (0, 0):
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x
        gx = numpy.zeros_like(y)
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    if sum_mode or inc_pad:
                        row_st = r * st0
                    else:
                        row_st = builtins.max(r * st0, self.padding[0])
                    row_end = builtins.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        if sum_mode or inc_pad:
                            col_st = c * st1
                        else:
                            col_st = builtins.max(c * st1,
                                                  self.padding[1])
                        col_end = builtins.min(col_st + ds1, img_cols)
                        if sum_mode:
                            val = gz[n, k, r, c]
                        else:
                            val = gz[n, k, r, c] / ((row_end - row_st) *
                                                    (col_end - col_st))
                        gx[n, k, row_st:row_end, col_st:col_end] += val
        # unpad the image
        gx = gx[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)]
        gx_stg[0] = gx

    def grad(self, inp, grads):
        x, gz = inp
        ggx, = grads
        return [theano.tensor.zeros_like(x),
                Pool(self.ds, ignore_border=self.ignore_border,
                     st=self.st, padding=self.padding, mode=self.mode)(ggx)]


class DownsampleFactorMaxGradGrad(OpenMPOp):
    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode')

    def __init__(self, ds, ignore_border, st=None, padding=(0, 0), mode='max', openmp=None):
        self.ds = tuple(ds)
        if not all([isinstance(d, integer_types) for d in ds]):
            raise ValueError(
                "Pool downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        assert isinstance(st, (tuple, list))
        self.st = tuple(st)
        self.ignore_border = ignore_border
        self.padding = tuple(padding)
        if self.padding != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
            raise NotImplementedError(
                'padding_h and padding_w must be smaller than strides')
        self.mode = mode
        super(DownsampleFactorMaxGradGrad, self).__init__(openmp=openmp)
        assert self.mode == 'max'

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of
        # MaxPoolGrad, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)
        assert x.ndim == 4
        assert maxout.ndim == 4
        assert gz.ndim == 4

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, inp, out):
        x, maxout, ggx = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'DownsampleFactorMaxGradGrad requires 4D input for now')
        if (z[0] is None) or (z[0].shape != maxout.shape):
            z[0] = numpy.zeros(maxout.shape, dtype=x.dtype)
        ggz = z[0]  # grad wrt maxout_grad has the same shape as maxout
        # number of pooling output rows
        pr = ggz.shape[-2]
        # number of pooling output cols
        pc = ggz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        img_rows = x.shape[-2] + 2 * pd0
        img_cols = x.shape[-1] + 2 * pd1

        # pad the image and its gradients
        if self.padding != (0, 0):
            y_padded = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype) + x.min() - 1
            y_padded[:, :, pd0:(img_rows - pd0), pd1:(img_cols - pd1)] = x
            ggx_padded = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            ggx_padded[:, :, pd0:(img_rows - pd0), pd1:(img_cols - pd1)] = ggx

        else:
            y_padded = x
            ggx_padded = ggx
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = builtins.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = builtins.min(col_st + ds1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == y_padded[n, k, row_ind, col_ind]):
                                    ggz[n, k, r, c] = ggx_padded[n, k, row_ind, col_ind]

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def c_code(self, node, name, inp, out, sub):
        if self.mode != 'max':
            raise theano.gof.utils.MethodNotDefined()
        x, maxout, ggx = inp
        z, = out  # the grad of grad
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, maximum) schedule(static)'
        else:
            omp_parallel = ''
        return """
        int z_typenum = PyArray_ObjectType((PyObject*)%(maxout)s, 0);
        int z_r, z_c;
        z_r = PyArray_DIMS(%(maxout)s)[2];
        z_c = PyArray_DIMS(%(maxout)s)[3];
        int r, c; // shape of the padded_input
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += %(pd0)s * 2;
        c += %(pd1)s * 2;
        // allocating memory for output
        if ((!%(z)s)
          || !PyArray_ISCONTIGUOUS(%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(maxout)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(maxout)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != PyArray_DIMS(%(maxout)s)[2])
          ||(PyArray_DIMS(%(z)s)[3] != PyArray_DIMS(%(maxout)s)[3])
          )
        {
          Py_XDECREF(%(z)s);
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, PyArray_DIMS(%(maxout)s), z_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(z)s, 0);
        }
        dtype_%(maxout)s maximum; // temp var for maximum value in a region
        int r_st, r_end, c_st, c_end;
        %(omp_parallel)s
        for(int t = 0; t < PyArray_DIMS(%(x)s)[0] * PyArray_DIMS(%(x)s)[1]; t++){
            int b = t %% PyArray_DIMS(%(x)s)[0];
            int k = t / PyArray_DIMS(%(x)s)[0];
                for(int i=0; i < z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    // from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // the maximum value
                    maximum = ((dtype_%(maxout)s*)(PyArray_GETPTR4(%(maxout)s,b,k,i,j)))[0];
                    // z at this position
                    dtype_%(z)s * z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        dtype_%(ggx)s * ggx = (
                          (dtype_%(ggx)s*)(PyArray_GETPTR4(%(ggx)s, b, k, m, n)));
                        if (a == maximum){
                          z[0] += ggx[0];
                        }
                      }
                    }
                  }
                }
              }
               printf(\"don poolGrad z:%%x\\n\",%(z)s);
        """ % locals()

    def c_code_cache_version(self):
        return (0, 1, self.openmp)

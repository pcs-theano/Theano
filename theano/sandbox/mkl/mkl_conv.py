"""
contains an op for convolving input images with a set of weights by using MKL
library, which is a free dnn library provided by Intel.
"""
from __future__ import absolute_import, print_function, division
# import logging
# from six import integer_types

import theano
from theano import Apply
from theano import gof
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
# from theano.tensor.blas import ldflags, blas_header_version
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import mkl_helper


class MKLConvBase(gof.Op):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'uniq_id')

    def __init__(self, imshp=None, kshp=None, border_mode="valid", subsample=(1, 1), uniq_id=0):
        if (not theano.config.blas.ldflags) or ('mkl' not in theano.config.blas.ldflags):
            raise NotImplementedError("MKL Convolution requires MKL library.")

        if isinstance(border_mode, int):
            if border_mode < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'non-negative integer'.format(border_mode))
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            if len(border_mode) != 2 or border_mode[0] < 0 or border_mode[1] < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'pair of non-negative integers'.format(border_mode))
            pad_h, pad_w = map(int, border_mode)
            border_mode = (pad_h, pad_w)
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(border_mode))
        self.border_mode = border_mode

        if len(subsample) != 2:
            raise ValueError("subsample must have two elements")
        self.subsample = tuple(subsample)
        self.imshp = imshp
        self.kshp = kshp
        self.uniq_id = uniq_id

    def __eq__(self, other):
        if hasattr(self, '__props__'):
            if type(self) != type(other):
                return False
            else:
                self_props = [getattr(self, p) for p in self.__props__ if p != 'uniq_id']
                other_props = [getattr(other, p) for p in other.__props__ if p != 'uniq_id']
                if self_props == other_props:
                    return True
                else:
                    return False
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.imshp) ^ hash(self. kshp) ^ hash(self.border_mode) ^ hash(self.subsample)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__, ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(MKLConvBase, self).c_compile_args()
        return compile_args

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_headers(self):
        headers = ['<stdio.h>']
        headers += super(MKLConvBase, self).c_headers()
        return headers

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (1, 0, self.uniq_id)

    def c_support_code(self):
        ccode = mkl_helper.header_text()
        ccode += """
            #define __DEBUG__ 0
            #define dimension 4
            #define CHECK_ERR(f, err) \\
                do { \\
                    (err) = (f); \\
                    if ((err) != E_SUCCESS) { \\
                        (PyExc_RuntimeError, "Error in file " \\
                            "[%s:%d], err code (%d)", __FILE__, __LINE__, err); \\
                    } \\
                } while(0)
        """
        return ccode

    def c_support_code_apply(self, node, name):
        dtype = node.inputs[0].dtype
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
            static int first_run = 1;
            static size_t bottomSize[dimension] = {0}; //w, h, c, n
            static size_t bottomStrides[dimension] = {0};
            static size_t weightSize[dimension+1] = {0}; //w, h, c, n
            static size_t weightStrides[dimension+1] = {0};
            static size_t topSize[dimension] = {0}; //w, h, c, n
            static size_t topStrides[dimension] = {0};
            static size_t groups = 1;
            static size_t fdimension = 0;

            ////////// debug only //////////
            static size_t bottom_size;
            static size_t weight_size;
            static size_t top_size;
            ///////////////////////////////

            static size_t convStrides[2] = {0};
            static int convPadding[2] = {0};

            static void *conv_res[dnnResourceNumber] = {0};

            static void *bottom_buffer_ptr = NULL;
            static void *bottom_buffer_ptr_from_previous = NULL;
            static void *bottom_buffer_ptr_to_previous = NULL;
            static void *weight_buffer_ptr = NULL;
            static void *top_buffer_ptr = NULL;
            static void *topgrad_buffer_ptr = NULL;

            static void *bwdf2fwd_weight_buffer_ptr = NULL;
            static void *weight_buffer_tmp_ptr = NULL;
            static void *topgrad_buffer_ptr_for_weight = NULL;

            static dnnError_t err;
            static dnnPrimitive_t pConvolutionFwd = NULL;
            static dnnPrimitive_t pConvolutionBwdData = NULL;
            static dnnPrimitive_t pConvolutionBwdFilter = NULL;

            static dnnPrimitive_t bwdf_convert_weight_to_fwd_int = NULL;
            static dnnPrimitive_t bwdf_convert_wegith_to_usr = NULL;
            static dnnPrimitive_t bwdd_convert_weight_to_bwdd_int = NULL;

            static dnnLayout_t bwdf_weight_int_layout = NULL;
            static dnnLayout_t bottom_usr_layout = NULL;
            static dnnLayout_t weight_usr_layout = NULL;
            static dnnLayout_t top_usr_layout = NULL;
            static dnnLayout_t bottom_int_layout = NULL;
            static dnnLayout_t *bottom_int_layout_ptr = NULL;
            static dnnLayout_t bottom_int_layout_from_previous = NULL;
            static dnnLayout_t weight_int_layout = NULL;
            static dnnLayout_t top_int_layout = NULL;
            static dnnLayout_t topgrad_int_layout = NULL;
            static dnnLayout_t topgrad_int_layout_for_weight = NULL;
            static dnnLayout_t fwd_weight_int_layout = NULL;

            static dnnPrimitive_t convert_bottom_to_int = NULL;
            static dnnPrimitive_t convert_weight_to_int = NULL;
            static dnnPrimitive_t convert_top_to_int = NULL;
            static dnnPrimitive_t convert_top_from_int = NULL;
            static dnnPrimitive_t convert_weight_from_int = NULL;
            static dnnPrimitive_t convert_bottom_from_int = NULL;
            static dnnPrimitive_t convert_int2int_bottom = NULL;
            static dnnPrimitive_t convert_int2int_topgrad_for_weight = NULL;
        """ % sub
        return ccode


class conv_forward(MKLConvBase):
    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=True, filter_dilation=(1, 1), uniq_id=0):
        super(conv_forward, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample, uniq_id=uniq_id)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"
        ccode = """
            //std::cout << "in c_cleanup_code_struct " << std::endl;
            //FIXME, remove below sentence if it's handled by conversion Op
            //dnnDelete_%(precision)s(convert_bottom_to_int);
            //dnnDelete_%(precision)s(convert_weight_to_int);
            //dnnDelete_%(precision)s(convert_top_to_int);
            //dnnDelete_%(precision)s(convert_top_from_int);
            //dnnDelete_%(precision)s(convert_weight_from_int);
            //dnnDelete_%(precision)s(convert_bottom_from_int);
            //dnnLayoutDelete_%(precision)s(bottom_usr_layout);
            //dnnLayoutDelete_%(precision)s(weight_usr_layout);
            //dnnLayoutDelete_%(precision)s(top_usr_layout);
            //dnnLayoutDelete_%(precision)s(bottom_int_layout);
            //dnnLayoutDelete_%(precision)s(weight_int_layout);
            //dnnLayoutDelete_%(precision)s(top_int_layout);
            //END
        """ % locals()
        return ccode

    def make_node(self, img, kern):
        img = as_tensor_variable(img)
        kern = as_tensor_variable(kern)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim not in [4, 5]:
            raise TypeError('kern must be 4D or 5D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        dtype = img.type.dtype
        return Apply(self, [img, kern], [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        imshp = input_shape[0]
        gkshp = input_shape[1]

        if len(gkshp) == 5:
            kshp = [gkshp[1] * gkshp[0], gkshp[2] * gkshp[0], gkshp[3], gkshp[4]]
        else:
            kshp = [gkshp[0], gkshp[1], gkshp[2], gkshp[3]]
        res = get_conv_output_shape(
            imshp,
            kshp,
            self.border_mode,
            self.subsample)
        return [res]

    def c_code(self, node, name, inp, out_, sub):
        bottom, weight, = inp
        top, = out_

        if self.imshp is None:
            imshp = node.inputs[0].shape
        else:
            imshp = self.imshp
        in_n, in_c, in_h, in_w = imshp

        if self.kshp is None:
            kshp = node.inputs[1].shape
        else:
            kshp = self.kshp

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = kshp
            assert in_c == k_c * grp
        else:
            k_n, k_c, k_h, k_w = kshp
            grp = 1
            assert in_c == k_c

        outshp = self.infer_shape(node, [imshp, kshp])
        o_n, o_c, o_h, o_w = outshp[0]

        dH, dW = self.subsample

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

        sub['bottom'] = bottom
        sub['weight'] = weight
        sub['top'] = top

        if node.inputs[0].dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'

        sub.update(locals())

        ccode = """
            #if __DEBUG__
                std::cout << "conv forward, c_code start" << std::endl;
            #endif
            if (NULL == pConvolutionFwd) {
                convStrides[0] = %(dW)s;
                convStrides[1] = %(dH)s;
                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                bottomSize[0] = %(in_w)s;  //w
                bottomSize[1] = %(in_h)s;  //h
                bottomSize[2] = %(in_c)s;  //c
                bottomSize[3] = %(in_n)s;  //n
                bottomStrides[0] = 1;
                bottomStrides[1] = bottomSize[0];
                bottomStrides[2] = bottomSize[0] * bottomSize[1];
                bottomStrides[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                weightSize[0] = %(k_w)s;
                weightSize[1] = %(k_h)s;
                weightSize[2] = %(k_c)s;
                weightSize[3] = %(k_n)s;
                weightSize[4] = %(grp)s;
                weightStrides[0] = 1;
                weightStrides[1] = weightSize[0];
                weightStrides[2] = weightSize[0] * weightSize[1];
                weightStrides[3] = weightSize[0] * weightSize[1] * weightSize[2];
                weightStrides[4] = weightSize[0] * weightSize[1] * weightSize[2] * weightSize[3];

                topSize[0] = %(o_w)s;
                topSize[1] = %(o_h)s;
                topSize[2] = %(o_c)s;
                topSize[3] = %(o_n)s;
                topStrides[0] = 1;
                topStrides[1] = topSize[0];
                topStrides[2] = topSize[0] * topSize[1];
                topStrides[3] = topSize[0] * topSize[1] * topSize[2];

                groups = %(grp)s;
                fdimension = dimension + (groups != 1);

                // Create conv forward primitive
                CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&pConvolutionFwd, NULL,
                           dnnAlgorithmConvolutionDirect, groups, dimension, bottomSize,
                           topSize, weightSize, convStrides, convPadding, dnnBorderZeros), err );
            }

            if (NULL == weight_usr_layout) {
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&weight_usr_layout, fdimension, weightSize, weightStrides), err );
            }
            if (NULL == bottom_int_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&bottom_int_layout,
                           pConvolutionFwd, dnnResourceSrc), err );
            }
            if (NULL == weight_int_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&weight_int_layout,
                           pConvolutionFwd, dnnResourceFilter), err );
            }
            if (NULL == top_int_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&top_int_layout,
                           pConvolutionFwd, dnnResourceDst), err );
            }

            // Prepare top array, only create once for passing internal layout and
            // internal data buffer for top data.
            if ( !(%(top)s && PyArray_NDIM(%(top)s) == 4)) {
               npy_intp out_dim[4];
               out_dim[0] = topSize[3];
               out_dim[1] = topSize[2];
               out_dim[2] = topSize[1];
               out_dim[3] = topSize[0];
               %(top)s = (PyArrayObject*)PyArray_ZEROS(dimension,
                                                        out_dim,
                                                        PyArray_TYPE(%(bottom)s),
                                                        0);
               if (NULL == %(top)s) {
                   PyErr_Format(PyExc_RuntimeError,
                                "conv_forward: Failed to allocate top of %%lld x %%lld x %%lld x %%lld",
                                (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3]);
                   %(fail)s
               }
            }

            #if __DEBUG__
                std::cout<<"bottom: "<<bottomSize[3]<<" x "<<bottomSize[2]<<" x "<<bottomSize[1]<<" x "<<bottomSize[0]<<std::endl;
                std::cout<<"weight: "<<weightSize[3]<<" x "<<weightSize[2]<<" x "<<weightSize[1]<<" x "<<weightSize[0]<<std::endl;
                std::cout<<"top: "<<topSize[3]<<" x "<<topSize[2]<<" x "<<topSize[1]<<" x "<<topSize[0]<<std::endl;
                std::cout<<"stride: "<<convStrides[1]<<" x "<<convStrides[0]<<std::endl;
                std::cout<<"padding: "<<convPadding[1]<<" x "<<convPadding[0]<<std::endl;
            #endif

            // get internal layout for input from previous Op
            bottom_int_layout_from_previous = ((dnnLayout_t*)PyArray_DATA(%(bottom)s))[0];
            // get internal buffer for input from previous op
            bottom_buffer_ptr_from_previous = ((void **)PyArray_DATA(%(bottom)s))[1];

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(bottom_int_layout_from_previous, bottom_int_layout)) {
                    #if __DEBUG__
                        std::cout<<"############bottom layout is not equal" <<std::endl;
                    #endif
                    if (NULL == convert_int2int_bottom) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_int2int_bottom, bottom_int_layout_from_previous, bottom_int_layout), err );
                    }
                }
            }
            if (convert_int2int_bottom) {
                if (NULL == bottom_buffer_ptr) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bottom_buffer_ptr, bottom_int_layout), err );
                }
                CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_bottom, bottom_buffer_ptr_from_previous, bottom_buffer_ptr), err );
                bottom_int_layout_ptr = &bottom_int_layout;
            } else {
                bottom_int_layout_ptr = &bottom_int_layout_from_previous;
                bottom_buffer_ptr = bottom_buffer_ptr_from_previous;
            }
            conv_res[dnnResourceSrc] = bottom_buffer_ptr;

            #if __DEBUG__
                std::cout<<"bottom internal layout = @"<<*bottom_int_layout_ptr<<std::endl;
                std::cout<<"bottom internal buffer = @"<<bottom_buffer_ptr<<std::endl;
            #endif

            weight_buffer_ptr = (%(dtype)s*)PyArray_DATA(%(weight)s);

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(weight_usr_layout, weight_int_layout)) {
                    if (NULL == convert_weight_to_int) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_weight_to_int, weight_usr_layout, weight_int_layout), err );
                    }
                }
            }

            #if __SUPPORT_USER_PARAMS__
                if (convert_weight_to_int) {
                    if (NULL == weight_buffer_tmp_ptr) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buffer_tmp_ptr, weight_int_layout), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(convert_weight_to_int, weight_buffer_ptr, weight_buffer_tmp_ptr), err );
                    conv_res[dnnResourceFilter] = weight_buffer_tmp_ptr;
                } else {
                    conv_res[dnnResourceFilter] = weight_buffer_ptr;
                }

            #else //__SUPPORT_USER_PARAMS__
                if (1 == first_run) {
                    if (convert_weight_to_int) {
                        if (NULL == weight_buffer_tmp_ptr) {
                            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buffer_tmp_ptr, weight_int_layout), err );
                        }
                        CHECK_ERR( dnnConversionExecute_%(precision)s(convert_weight_to_int, weight_buffer_ptr, weight_buffer_tmp_ptr), err );
                        memcpy(weight_buffer_ptr, weight_buffer_tmp_ptr, dnnLayoutGetMemorySize_%(precision)s(weight_int_layout));
                    }
                }
                conv_res[dnnResourceFilter] = weight_buffer_ptr;
            #endif //__SUPPORT_USER_PARAMS__

            //Allocate internal buffer for top data, only apply once
            if (NULL == top_buffer_ptr) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&top_buffer_ptr, top_int_layout), err );
            }
            conv_res[dnnResourceDst] = top_buffer_ptr;

            #if __DEBUG__
                bottom_size = dnnLayoutGetMemorySize_%(precision)s(*bottom_int_layout_ptr);
                weight_size = dnnLayoutGetMemorySize_%(precision)s(weight_int_layout);
                top_size = dnnLayoutGetMemorySize_%(precision)s(top_int_layout);
                std::cout << "forward, pConvolution = @" << pConvolutionFwd << std::endl;
                std::cout<<"bottom size: "<<bottomSize[3]<<" x "<<bottomSize[2]<<" x "<<bottomSize[1]<<" x "<<bottomSize[0]<<", acutal size: "<<bottom_size<<std::endl;
                std::cout<<"bottom buffer ptr: "<<bottom_buffer_ptr<<std::endl;
                std::cout<<"weight size: "<<weightSize[3]<<" x "<<weightSize[2]<<" x "<<weightSize[1]<<" x "<<weightSize[0]<<", actual size: "<<weight_size<<std::endl;
                std::cout<<"weight buffer ptr: "<<weight_buffer_ptr<<std::endl;
                std::cout<<"top size: "<<topSize[3]<<" x "<<topSize[2]<<" x "<<topSize[1]<<" x "<<topSize[0]<<", actual size: "<<top_size<<std::endl;
                std::cout<<"top buffer ptr: "<<top_buffer_ptr<<std::endl;
                std::cout << "forward, pConvolution = @" << pConvolutionFwd << std::endl;
                std::cout << "forward, conv_res[Src] = @" << conv_res[dnnResourceSrc] << std::endl;
                std::cout << "forward, conv_res[Filter] = @" << conv_res[dnnResourceFilter] << std::endl;
                std::cout << "forward, conv_res[Dst] = @" << conv_res[dnnResourceDst] << std::endl;
            #endif

            //Execute convolution forward pass
            CHECK_ERR( dnnExecute_%(precision)s(pConvolutionFwd, (void**)conv_res), err );

            // return the internal layout and data buffer for top directly.
            ((dnnLayout_t*)PyArray_DATA(%(top)s))[0] = top_int_layout;
            ((void**)PyArray_DATA(%(top)s))[1] = top_buffer_ptr;

            first_run = 0;
            #if __DEBUG__
                printf(\"convFw z:%%x, %%x, %%x\\n\",%(top)s,top_int_layout,top_buffer_ptr);
                std::cout <<"conv forward, top_int_layout: @" <<top_int_layout<<std::endl;
                std::cout <<"conv forward, top_buffer_ptr: @" <<top_buffer_ptr<<std::endl;
                std::cout << "forward, c_code end\\n" << std::endl;
            #endif
        """ % sub
        return ccode

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        d_image = conv_gradInputs(border_mode=self.border_mode,
                                  subsample=self.subsample,
                                  imshp=self.imshp,
                                  kshp=self.kshp)(bottom, weights, top)
        d_weights = conv_gradWeights(border_mode=self.border_mode,
                                     subsample=self.subsample,
                                     imshp=self.imshp,
                                     kshp=self.kshp)(bottom, weights, top)
        return d_image, d_weights


class conv_gradInputs(MKLConvBase):
    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=True, filter_dilation=(1, 1), uniq_id=0):
        super(conv_gradInputs, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample, uniq_id=uniq_id)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"
        ccode = """
            //std::cout << "in gradI c_cleanup_code_struct " << std::endl;
            //FIXME, remove below sentence if it's handled by conversion Op
            //dnnDelete_%(precision)s(convert_bottom_to_int);
            //dnnDelete_%(precision)s(convert_weight_to_int);
            //dnnDelete_%(precision)s(convert_top_to_int);
            //dnnDelete_%(precision)s(convert_top_from_int);
            //dnnDelete_%(precision)s(convert_weight_from_int);
            //dnnDelete_%(precision)s(convert_bottom_from_int);
            //dnnLayoutDelete_%(precision)s(bottom_usr_layout);
            //dnnLayoutDelete_%(precision)s(weight_usr_layout);
            //dnnLayoutDelete_%(precision)s(top_usr_layout);
            //dnnLayoutDelete_%(precision)s(bottom_int_layout);
            //dnnLayoutDelete_%(precision)s(weight_int_layout);
            //dnnLayoutDelete_%(precision)s(top_int_layout);
            //END
        """ % locals()
        return ccode

    def make_node(self, image, kern, topgrad):
        image = as_tensor_variable(image)
        kern = as_tensor_variable(kern)
        topgrad = as_tensor_variable(topgrad)
        if kern.type.ndim not in [4, 5]:
            raise TypeError('kern must be 4D or 5D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')

        broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[1],
                         False, False]
        dtype = kern.type.dtype
        return Apply(self, [image, kern, topgrad], [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        imshp = input_shape[0]
        gkshp = input_shape[1]

        if len(gkshp) == 5:
            kshp = [gkshp[1] * gkshp[0], gkshp[2] * gkshp[0], gkshp[3], gkshp[4]]
        else:
            kshp = [gkshp[0], gkshp[1], gkshp[2], gkshp[3]]
        res = get_conv_output_shape(imshp, kshp, self.border_mode, self.subsample)
        return [res]

    def c_code(self, node, name, inp, out_, sub):
        bottom, weights, topgrad = inp
        bottomgrad, = out_
        if self.imshp is None:
            imshp = node.inputs[0].shape
        else:
            imshp = self.imshp
        in_n, in_c, in_h, in_w = imshp

        if self.kshp is None:
            kshp = node.inputs[1].shape
        else:
            kshp = self.kshp

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = kshp
            assert in_c == k_c * grp
        else:
            grp = 1
            k_n, k_c, k_h, k_w = kshp
            assert in_c == k_c

        if node.inputs[1].type.ndim == 5:
            tshp = [kshp[1] * kshp[0], kshp[2] * kshp[0], kshp[3], kshp[4]]
        else:
            tshp = [kshp[0], kshp[1], kshp[2], kshp[3]]

        outshp = get_conv_output_shape(imshp, tshp, self.border_mode, self.subsample)

        o_n, o_c, o_h, o_w = outshp

        dH, dW = self.subsample

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

        sub['bottom'] = bottom
        sub['bottomgrad'] = bottomgrad
        sub['weight'] = weights
        sub['topgrad'] = topgrad

        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'
        sub.update(locals())

        ccode = """
            #if __DEBUG__
                std::cout << "gradInput, c_code start " << std::endl;
            #endif
            if (NULL == pConvolutionBwdData) {
                convStrides[0] = %(dW)s;
                convStrides[1] = %(dH)s;

                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                bottomSize[0] = %(in_w)s;  //w
                bottomSize[1] = %(in_h)s;  //h
                bottomSize[2] = %(in_c)s;  //c
                bottomSize[3] = %(in_n)s;  //n
                bottomStrides[0] = 1;
                bottomStrides[1] = bottomSize[0];
                bottomStrides[2] = bottomSize[0] * bottomSize[1];
                bottomStrides[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                weightSize[0] = %(k_w)s;
                weightSize[1] = %(k_h)s;
                weightSize[2] = %(k_c)s;
                weightSize[3] = %(k_n)s;
                weightSize[4] = %(grp)s;
                weightStrides[0] = 1;
                weightStrides[1] = weightSize[0];
                weightStrides[2] = weightSize[0] * weightSize[1];
                weightStrides[3] = weightSize[0] * weightSize[1] * weightSize[2];
                weightStrides[4] = weightSize[0] * weightSize[1] * weightSize[2] * weightSize[3];

                topSize[0] = %(o_w)s;
                topSize[1] = %(o_h)s;
                topSize[2] = %(o_c)s;
                topSize[3] = %(o_n)s;
                topStrides[0] = 1;
                topStrides[1] = topSize[0];
                topStrides[2] = topSize[0] * topSize[1];
                topStrides[3] = topSize[0] * topSize[1] * topSize[2];

                groups = %(grp)s;
                fdimension = dimension + (groups != 1);

                // Create conv gradInput primitive
                CHECK_ERR( dnnGroupsConvolutionCreateBackwardData_%(precision)s(&pConvolutionBwdData, NULL,
                           dnnAlgorithmConvolutionDirect, groups, dimension, bottomSize,
                           topSize, weightSize, convStrides, convPadding, dnnBorderZeros), err );
            }
            if (NULL == weight_int_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&weight_int_layout,
                           pConvolutionBwdData, dnnResourceFilter), err );
            }
            if (NULL == bottom_int_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&bottom_int_layout,
                           pConvolutionBwdData, dnnResourceDiffSrc), err );
            }

            if (NULL == pConvolutionFwd) {
                // Create conv forward primitive
                CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&pConvolutionFwd, NULL,
                           dnnAlgorithmConvolutionDirect, groups, dimension, bottomSize,
                           topSize, weightSize, convStrides, convPadding, dnnBorderZeros), err );
            }
            if(NULL == fwd_weight_int_layout) {
                CHECK_ERR(dnnLayoutCreateFromPrimitive_%(precision)s(&fwd_weight_int_layout,
                          pConvolutionFwd, dnnResourceFilter), err );
            }

            if ( !(%(bottomgrad)s)) {
                %(bottomgrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(bottom)s),
                                                               PyArray_DIMS(%(bottom)s),
                                                               PyArray_TYPE(%(bottom)s),
                                                               0);
                if (NULL == %(bottomgrad)s) {
                    PyErr_Format(PyExc_RuntimeError,
                                "conv_gradInput: Failed to allocate bottom of %%lld x %%lld x %%lld x %%lld",
                                (long long)(PyArray_DIMS(%(bottom)s))[0], (long long)(PyArray_DIMS(%(bottom)s))[1],
                                (long long)(PyArray_DIMS(%(bottom)s))[2], (long long)(PyArray_DIMS(%(bottom)s))[3]);
                    %(fail)s
                }
           }

           //weight use its own buffer
           weight_buffer_ptr = (%(dtype)s*)PyArray_DATA(%(weight)s);

           //get internal layout for topgrad from previous Op
           topgrad_int_layout = ((dnnLayout_t*)PyArray_DATA(%(topgrad)s))[0];
           //get internal buffer for topgrad from previous op
           topgrad_buffer_ptr = ((void **)PyArray_DATA(%(topgrad)s))[1];

           conv_res[dnnResourceDiffDst] = topgrad_buffer_ptr;

           #if __SUPPORT_USER_PARAMS__
               if(NULL == weight_usr_layout) {
                   CHECK_ERR( dnnLayoutCreate_%(precision)s(&weight_usr_layout, fdimension, weightSize, weightStrides), err );
               }

               if (1 == first_run) {
                   if (!dnnLayoutCompare_%(precision)s(weight_usr_layout, weight_int_layout)) {
                       if(NULL == convert_weight_to_int) {
                           CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_weight_to_int, weight_usr_layout, weight_int_layout), err );
                       }
                   }
               }

               if (convert_weight_to_int) {
                   if(NULL == weight_buffer_tmp_ptr) {
                       CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buffer_tmp_ptr, weight_int_layout), err );
                   }
                   CHECK_ERR( dnnConversionExecute_%(precision)s(convert_weight_to_int, weight_buffer_ptr, weight_buffer_tmp_ptr), err );
               } else {
                   weight_buffer_tmp_ptr = weight_buffer_ptr;
               }
           #else
               if (1 == first_run) {
                   if (!dnnLayoutCompare_%(precision)s(fwd_weight_int_layout, weight_int_layout)) {
                       if(NULL == bwdd_convert_weight_to_bwdd_int) {
                           CHECK_ERR( dnnConversionCreate_%(precision)s(&bwdd_convert_weight_to_bwdd_int, fwd_weight_int_layout, weight_int_layout), err );
                       }
                   }
               }
               if (bwdd_convert_weight_to_bwdd_int) {
                   if(NULL == weight_buffer_tmp_ptr) {
                       CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buffer_tmp_ptr, weight_int_layout), err );
                   }
                   CHECK_ERR( dnnConversionExecute_%(precision)s(bwdd_convert_weight_to_bwdd_int, weight_buffer_ptr, weight_buffer_tmp_ptr), err );
               } else {
                   weight_buffer_tmp_ptr = weight_buffer_ptr;
               }
           #endif

           conv_res[dnnResourceFilter] = weight_buffer_tmp_ptr;

           //Allocate internal buffer for bottomgrad data
           if (NULL == bottom_buffer_ptr) {
               CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bottom_buffer_ptr, bottom_int_layout), err );
           }
           conv_res[dnnResourceDiffSrc] = bottom_buffer_ptr;

           //Execute convolution gradInput pass
           CHECK_ERR( dnnExecute_%(precision)s(pConvolutionBwdData, (void**)conv_res), err );

           //get bottom_int_layout from forward pass, pass the data buffer match previous layout.
           bottom_int_layout_from_previous = ((dnnLayout_t*)PyArray_DATA(%(bottom)s))[0];

           //bottom int2int cvt
           if (1 == first_run) {
               if (!dnnLayoutCompare_%(precision)s(bottom_int_layout, bottom_int_layout_from_previous)) {
                   #if __DEBUG__
                       std::cout<<"############gradInput, input layout is not equal" <<std::endl;
                   #endif
                   if (NULL == convert_int2int_bottom) {
                       CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_int2int_bottom, bottom_int_layout, bottom_int_layout_from_previous), err );
                   }
               }
           }
           if (convert_int2int_bottom) {
               if (NULL == bottom_buffer_ptr_to_previous) {
                   CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bottom_buffer_ptr_to_previous, bottom_int_layout_from_previous), err );
               }
               CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_bottom, bottom_buffer_ptr, bottom_buffer_ptr_to_previous), err );
           } else {
               bottom_buffer_ptr_to_previous = bottom_buffer_ptr;
           }

           ((dnnLayout_t*)PyArray_DATA(%(bottomgrad)s))[0] = bottom_int_layout_from_previous;
           ((void**)PyArray_DATA(%(bottomgrad)s))[1] = bottom_buffer_ptr_to_previous;

           first_run = 0;

           #if __DEBUG__
               size_t bottom_size = dnnLayoutGetMemorySize_%(precision)s(bottom_int_layout);
               std::cout << "gradInput, bottom_size: "<<bottom_size<<std::endl;
               std::cout << "gradInput, c_code end\\n" << std::endl;
           #endif
        """ % sub
        return ccode


class conv_gradWeights(MKLConvBase):
    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=True, filter_dilation=(1, 1), uniq_id=0):
        super(conv_gradWeights, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample, uniq_id=uniq_id)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"
        uniq = self.uniq_id
        ccode = """
            //std::cout << "in gradW c_cleanup_code_struct " << std::endl;
            //FIXME, remove below sentence if it's handled by conversion Op
            //dnnDelete_%(precision)s(convert_bottom_to_int);
            //dnnDelete_%(precision)s(convert_weight_to_int);
            //dnnDelete_%(precision)s(convert_top_to_int);
            //dnnDelete_%(precision)s(convert_top_from_int);
            //dnnDelete_%(precision)s(convert_weight_from_int);
            //dnnDelete_%(precision)s(convert_bottom_from_int);
            //dnnLayoutDelete_%(precision)s(bottom_usr_layout);
            //dnnLayoutDelete_%(precision)s(weight_usr_layout);
            //dnnLayoutDelete_%(precision)s(top_usr_layout);
            //dnnLayoutDelete_%(precision)s(bottom_int_layout);
            //dnnLayoutDelete_%(precision)s(weight_int_layout);
            //dnnLayoutDelete_%(precision)s(top_int_layout);
            //END
            // conv_%(uniq)s
        """ % locals()
        return ccode

    def make_node(self, image, weight, topgrad):
        image = as_tensor_variable(image)
        weight = as_tensor_variable(weight)
        topgrad = as_tensor_variable(topgrad)
        if image.type.ndim != 4:
            raise TypeError('image must be 4D tensor')
        if weight.type.ndim not in [4, 5]:
            raise TypeError('weightmust be 4D or 5D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')

        # broadcastable = [topgrad.type.broadcastable[1], image.type.broadcastable[1],
        #                  False, False]
        # dtype = image.type.dtype
        return Apply(self, [image, weight, topgrad], [weight.type()])

    def c_code(self, node, name, inp, out_, sub):
        bottom, weight, topgrad = inp
        weightgrad, = out_

        if self.imshp is None:
            imshp = node.inputs[0].shape
        else:
            imshp = self.imshp
        in_n, in_c, in_h, in_w = imshp

        if self.kshp is None:
            kshp = node.inputs[1].shape
        else:
            kshp = self.kshp

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = kshp
            tshp = [kshp[1] * kshp[0], kshp[2] * kshp[0], kshp[3], kshp[4]]
            assert in_c == k_c * grp
        else:
            k_n, k_c, k_h, k_w = kshp
            grp = 1
            tshp = [kshp[0], kshp[1], kshp[2], kshp[3]]
            assert in_c == k_c

        outshp = get_conv_output_shape(imshp, tshp, self.border_mode, self.subsample)

        o_n, o_c, o_h, o_w = outshp

        dH, dW = self.subsample
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

        sub['bottom'] = bottom
        sub['weight'] = weight
        sub['weightgrad'] = weightgrad
        sub['topgrad'] = topgrad

        if node.inputs[0].dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'
        sub.update(locals())

        ccode = """
            ////bwdfilter related
            if (NULL == pConvolutionBwdFilter) {
                convStrides[0] = %(dW)s;
                convStrides[1] = %(dH)s;
                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                bottomSize[0] = %(in_w)s;  //w
                bottomSize[1] = %(in_h)s;  //h
                bottomSize[2] = %(in_c)s;  //c
                bottomSize[3] = %(in_n)s;  //n
                bottomStrides[0] = 1;
                bottomStrides[1] = bottomSize[0];
                bottomStrides[2] = bottomSize[0] * bottomSize[1];
                bottomStrides[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                weightSize[0] = %(k_w)s;
                weightSize[1] = %(k_h)s;
                weightSize[2] = %(k_c)s;
                weightSize[3] = %(k_n)s;
                weightSize[4] = %(grp)s;
                weightStrides[0] = 1;
                weightStrides[1] = weightSize[0];
                weightStrides[2] = weightSize[0] * weightSize[1];
                weightStrides[3] = weightSize[0] * weightSize[1] * weightSize[2];
                weightStrides[4] = weightSize[0] * weightSize[1] * weightSize[2] * weightSize[3];
                topSize[0] = %(o_w)s;
                topSize[1] = %(o_h)s;
                topSize[2] = %(o_c)s;
                topSize[3] = %(o_n)s;
                topStrides[0] = 1;
                topStrides[1] = topSize[0];
                topStrides[2] = topSize[0] * topSize[1];
                topStrides[3] = topSize[0] * topSize[1] * topSize[2];

                groups = %(grp)s;
                fdimension = dimension + (groups != 1);

                // Create conv backward primitive
                CHECK_ERR( dnnGroupsConvolutionCreateBackwardFilter_%(precision)s(&pConvolutionBwdFilter, NULL,
                           dnnAlgorithmConvolutionDirect, groups, dimension, bottomSize,
                           topSize, weightSize, convStrides, convPadding, dnnBorderZeros), err );
            }
            if (NULL == bwdf_weight_int_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&bwdf_weight_int_layout,
                           pConvolutionBwdFilter, dnnResourceDiffFilter), err );
            }

            if (NULL == bottom_int_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&bottom_int_layout,
                           pConvolutionBwdFilter, dnnResourceSrc), err );
            }

            if (NULL == topgrad_int_layout_for_weight) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&topgrad_int_layout_for_weight,
                           pConvolutionBwdFilter, dnnResourceDiffDst), err );
            }

            // create forward primitive here to get forward internal layout
            if (NULL == pConvolutionFwd) {
                CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&pConvolutionFwd, NULL,
                           dnnAlgorithmConvolutionDirect, groups, dimension, bottomSize,
                           topSize, weightSize, convStrides, convPadding, dnnBorderZeros), err );
            }

            if (NULL == fwd_weight_int_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&fwd_weight_int_layout,
                           pConvolutionFwd, dnnResourceFilter), err );
            }

            //// Prepare weightgrad array
            if ( !(%(weightgrad)s) ) {
                %(weightgrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(weight)s),
                                                               PyArray_DIMS(%(weight)s),
                                                               PyArray_TYPE(%(weight)s),
                                                               0);
                if (NULL == %(weightgrad)s) {
                    PyErr_Format(PyExc_RuntimeError,
                            "conv_gradWeight: Failed to allocate weight of %%lld x %%lld x %%lld x %%lld x %%lld",
                            (long long)(PyArray_DIMS(%(weight)s))[0], (long long)(PyArray_DIMS(%(weight)s))[1],
                            (long long)(PyArray_DIMS(%(weight)s))[2], (long long)(PyArray_DIMS(%(weight)s))[3]);
                }
            }

            weight_buffer_ptr = (%(dtype)s*)PyArray_DATA(%(weightgrad)s);

            // get internal layout for input from previous Op
            bottom_int_layout_from_previous = ((dnnLayout_t*)PyArray_DATA(%(bottom)s))[0];
            // get internal buffer for input from previous op
            bottom_buffer_ptr_from_previous = ((void **)PyArray_DATA(%(bottom)s))[1];

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(bottom_int_layout_from_previous, bottom_int_layout)) {
                    #if __DEBUG__
                        std::cout<<"############gradWeight, bottom layout is not equal" <<std::endl;
                    #endif
                    if (NULL == convert_int2int_bottom) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_int2int_bottom, bottom_int_layout_from_previous, bottom_int_layout), err );
                    }
                }
            }

            if (convert_int2int_bottom) {
                if (NULL == bottom_buffer_ptr) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bottom_buffer_ptr, bottom_int_layout), err );
                }
                CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_bottom, bottom_buffer_ptr_from_previous, bottom_buffer_ptr), err );
                bottom_int_layout_ptr = &bottom_int_layout;
            } else {
                bottom_int_layout_ptr = &bottom_int_layout_from_previous;
                bottom_buffer_ptr = bottom_buffer_ptr_from_previous;
            }

            // get internal layout for topgrad from previous Op
            topgrad_int_layout = ((dnnLayout_t*)PyArray_DATA(%(topgrad)s))[0];
            // get internal buffer for topgrad from previous op
            topgrad_buffer_ptr = ((void **)PyArray_DATA(%(topgrad)s))[1];

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(topgrad_int_layout, topgrad_int_layout_for_weight)) {
                    #if __DEBUG__
                        std::cout<<"############gradWeight, topgrad layout is not equal for weight" <<std::endl;
                    #endif
                    if (NULL == convert_int2int_topgrad_for_weight) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_int2int_topgrad_for_weight, topgrad_int_layout, topgrad_int_layout_for_weight), err );
                    }
                }
            }
            if (convert_int2int_topgrad_for_weight) {
                if (NULL == topgrad_buffer_ptr_for_weight) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&topgrad_buffer_ptr_for_weight, topgrad_int_layout_for_weight), err );
                }
                CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_topgrad_for_weight, topgrad_buffer_ptr, topgrad_buffer_ptr_for_weight), err );
            } else {
                topgrad_buffer_ptr_for_weight = topgrad_buffer_ptr;
            }

            conv_res[dnnResourceSrc] = bottom_buffer_ptr;
            conv_res[dnnResourceDiffDst] = topgrad_buffer_ptr_for_weight;

            //Allocate internal buffer for weightgrad data
            if (NULL == weight_buffer_tmp_ptr) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buffer_tmp_ptr, bwdf_weight_int_layout), err );
            }
            conv_res[dnnResourceDiffFilter] = weight_buffer_tmp_ptr;

            //Execute convolution gradweight pass
            CHECK_ERR( dnnExecute_%(precision)s(pConvolutionBwdFilter, (void**)conv_res), err );

            //weight bwd -> fwd cvt
            if(!dnnLayoutCompare_%(precision)s(bwdf_weight_int_layout, fwd_weight_int_layout)) {
                #if __DEBUG__
                    std::cout<<"############gradWeight, bwdf_weight_int_layout is not equal to fwd_weight_int_layout" <<std::endl;
                #endif
                if (NULL == bwdf_convert_weight_to_fwd_int) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&bwdf_convert_weight_to_fwd_int, bwdf_weight_int_layout, fwd_weight_int_layout), err );
                }
            }

            #if __SUPPORT_USER_PARAMS__
                if(NULL == weight_usr_layout) {
                    dnnLayoutCreate_%(precision)s(&weight_usr_layout, fdimension, weightSize, weightStrides);
                    //printf(\"gradweight, weightstride: %%d, %%d, %%d, %%d\\n\", weightStrides[0], weightStrides[1], weightStrides[2], weightStrides[3]);
                }

                if (bwdf_convert_weight_to_fwd_int) {
                    if (NULL == bwdf2fwd_weight_buffer_ptr) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bwdf2fwd_weight_buffer_ptr, fwd_weight_int_layout), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(bwdf_convert_weight_to_fwd_int, weight_buffer_tmp_ptr, bwdf2fwd_weight_buffer_ptr), err );
                } else {
                    bwdf2fwd_weight_buffer_ptr = weight_buffer_tmp_ptr;
                }

                if (NULL == bwdf_convert_wegith_to_usr) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&bwdf_convert_wegith_to_usr, fwd_weight_int_layout, weight_usr_layout), err );
                }
                dnnConversionExecute_%(precision)s(bwdf_convert_wegith_to_usr, bwdf2fwd_weight_buffer_ptr, weight_buffer_ptr);
            #else
                if (bwdf_convert_weight_to_fwd_int) {
                    if (NULL == bwdf2fwd_weight_buffer_ptr) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bwdf2fwd_weight_buffer_ptr, fwd_weight_int_layout), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(bwdf_convert_weight_to_fwd_int, weight_buffer_tmp_ptr, weight_buffer_ptr), err );
                }
                else {
                    memcpy(weight_buffer_ptr, weight_buffer_tmp_ptr, dnnLayoutGetMemorySize_%(precision)s(fwd_weight_int_layout));
                }
            #endif  //__SUPPORT_USER_PARAMS__

            first_run = 0;

            #if __DEBUG__
                std::cout << "gradWeight, c_code end\\n" << std::endl;
            #endif
        """ % sub
        return ccode

from theano import Variable, Apply
from theano.tensor import as_tensor_variable
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import mkl_helper, basic_ops


class Relu(basic_ops.MKLOp):
    __props__ = ('slope', 'uniq_id')

    def __init__(self, slope=0, uniq_id=0):
        self.slope = slope
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
        return hash(self.slope)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ','.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [ReluGrad(slope=self.slope, uniq_id=self.uniq_id)(x, gz)]

    def c_support_code(self):
        return mkl_helper.header_text() + """
        static int first_run = 1;
        static int count = 0;
        static int typenum;
        static int x_bs;
        static int x_channels;
        static int x_row;
        static int x_col;
        static dnnPrimitive_t batchNormBwdData;
        static dnnPrimitive_t batchNormBwdScaleShift;
        static size_t dim = 4;
        static size_t sizes[4];
        static size_t strides[4];
        static dnnError_t e;
        static dnnLayout_t *fwd_bottom_data_int_l_p;
        static dnnLayout_t fwd_bottom_data_int_l;
        static dnnLayout_t fwd_bottom_data_usr_l=NULL;
        static dnnPrimitive_t fwd_bottom_convert_to_int=NULL;
        static dnnPrimitive_t fwd_bottom_convert_from_int=NULL;
        static dnnPrimitive_t fwd_bottom_convert_prv2prv=NULL;

        static dnnLayout_t fwd_top_data_usr_l=NULL;
        static dnnLayout_t fwd_top_data_int_l=NULL;
        static dnnPrimitive_t fwd_top_convert_to_int=NULL;
        static dnnPrimitive_t fwd_top_convert_from_int=NULL;
        static dnnPrimitive_t fwd_top_convert_prv2prv=NULL;
        static dnnPrimitive_t reluFwd  = static_cast<dnnPrimitive_t>(NULL);
        static void* relu_res[dnnResourceNumber];
        static void* output_buffer_ptr=NULL;
        static void* input_buffer_ptr=NULL;
        static int xy=0x10;
        static PyArrayObject* real_buffer = NULL;
        static dnnLayout_t* layout_p=NULL;
        #define L_PASS (1)
        #define __DEBUG__ 0
        """

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
        else:
            raise TypeError('input must be float32 or float64')

        return """
        dnnReleaseBuffer_%(precision)s(buffer);
        """ % sub

    def c_headers(self):
        return ['<math.h>', '<iostream>']

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        slope = self.slope

        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
        else:
            raise TypeError('input must be float32 or float64')

        sub = sub.copy()
        sub.update(locals())

        ret = """
        {
            #if __DEBUG__
            std::cout<<"Relu Fwd start"<<std::endl;
            #endif
            if(first_run){
                x_bs = PyArray_DIMS(%(x)s)[0];
                x_channels = PyArray_DIMS(%(x)s)[1];
                x_row = PyArray_DIMS(%(x)s)[2];
                x_col = PyArray_DIMS(%(x)s)[3];
                sizes[0] = x_col;
                sizes[1] = x_row;
                sizes[2] = x_channels;
                sizes[3] = x_bs;
                strides[0] = 1;
                strides[1] = sizes[0];
                strides[2] = sizes[0]*sizes[1];
                strides[3] = sizes[0]*sizes[1]*sizes[2];
            }

            npy_intp dims[4] = {0, 0, 0, 0};
            dims[0] = PyArray_DIMS(%(x)s)[0];
            dims[1] = PyArray_DIMS(%(x)s)[1];
            dims[2] = PyArray_DIMS(%(x)s)[2];
            dims[3] = PyArray_DIMS(%(x)s)[3];
            if ( !(%(z)s
                    && PyArray_NDIM(%(z)s) == 4
                    && PyArray_IS_C_CONTIGUOUS(%(z)s)
                    && PyArray_DIMS(%(z)s)[0] == dims[0]
                    && PyArray_DIMS(%(z)s)[1] == dims[1]
                    && PyArray_DIMS(%(z)s)[2] == dims[2]
                    && PyArray_DIMS(%(z)s)[3] == dims[3])) {
                Py_XDECREF(%(z)s);
                typenum = PyArray_TYPE(%(x)s);
                %(z)s = (PyArrayObject*)PyArray_ZEROS(4,
                                                  dims,
                                                  typenum,
                                                  0);
                if (NULL == %(z)s) {
                    PyErr_Format(PyExc_RuntimeError,
                                "relu forward: Failed to allocate output of %%lld x %%lld x %%lld x %%lld",
                                (long long)dims[0], (long long)dims[1], (long long)dims[2], (long long)dims[3]);
                    %(fail)s
                }
            }

            //std::cout<<"start fwd bs "<<x_bs<<" channel "<<x_channels<<" row "<<x_row<<" col "<<x_col<<std::endl;

            #ifdef USER_LAYOUT
            if(first_run){
                e = dnnLayoutCreate_%(precision)s(&fwd_bottom_data_usr_l, dim, sizes, strides);
                if (E_SUCCESS != e){
                  std::cout<<"relu fwd_bottom_data_usr_l creat fail with error code "<<e<<std::endl;
                }
                e = dnnLayoutCreate_%(precision)s(&fwd_top_data_usr_l, dim, sizes, strides);
                if (E_SUCCESS != e){
                  std::cout<<"relu fwd_top_data_usr_l creat fail\\n";
                }

                e = dnnReLUCreateForward_%(precision)s(&reluFwd, NULL, fwd_bottom_data_usr_l, %(slope)s);
                if (E_SUCCESS != e){
                    std::cout<<"relu fwd creat fail with error code "<<e<<std::endl;
                }
            #endif

            //get internal layout and buffer from previous Op
            fwd_bottom_data_int_l = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];
            input_buffer_ptr = ((void **)PyArray_DATA(%(x)s))[1];

            if (first_run) {
                e = dnnReLUCreateForward_%(precision)s(&reluFwd, NULL, fwd_bottom_data_int_l, %(slope)s);
                if (E_SUCCESS != e){
                    std::cout<<"relu fwd creat fail with error code "<<e<<std::endl;
                }

                e = dnnLayoutCreateFromPrimitive_%(precision)s(&fwd_top_data_int_l, reluFwd, dnnResourceDst);
                if (E_SUCCESS != e){
                  std::cout<<"relu fwd_top_data_int_l creat fail with error code "<<e<<std::endl;
                }
            }

            #if __DEBUG__
                std::cout<<"relu forward: fwd_bottom_data_int_l:"<<fwd_bottom_data_int_l<<std::endl;
                std::cout<<"relu forward: input:"<<input_buffer_ptr<<std::endl;
                size_t img_size = dnnLayoutGetMemorySize_%(precision)s(fwd_bottom_data_int_l);
                size_t out_size = dnnLayoutGetMemorySize_%(precision)s(fwd_top_data_int_l);
                std::cout<<"input size: "<<sizes[3]<<" x "<<sizes[2]<<" x "<<sizes[1]<<" x "<<sizes[0]<<std::endl;
                std::cout<<"relu forward: input size in bytes: "<<img_size<<", output size in bytes: "<<out_size<<std::endl;
            #endif
            if (NULL == output_buffer_ptr) {
                e = dnnAllocateBuffer_%(precision)s(&output_buffer_ptr, fwd_top_data_int_l);
                if (E_SUCCESS != e){
                  std::cout<<"relu allocate fail with error code "<<e<<std::endl;
                }
                memset(output_buffer_ptr,0,dnnLayoutGetMemorySize_%(precision)s(fwd_top_data_int_l));
            }

            relu_res[dnnResourceDst] = (void*)output_buffer_ptr;
            relu_res[dnnResourceSrc] = (void*)input_buffer_ptr;


            if (E_SUCCESS != dnnExecute_%(precision)s(reluFwd, relu_res)){
              std::cout<<"relu fwd execute fail"<<std::endl;
            }

            ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = fwd_top_data_int_l;
            ((void**)PyArray_DATA(%(z)s))[1] = output_buffer_ptr;

            first_run = 0;

            #if __DEBUG__
            std::cout<<"relu forward: output layout=@"<<fwd_top_data_int_l<<", output_buffer_ptr=@"<<output_buffer_ptr<<std::endl;
            std::cout<<"relu fwd finished\\n"<<std::endl;
            #endif
        }
        """ % sub
        return ret

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)


class ReluGrad(basic_ops.MKLOp):
    __props__ = ('slope', 'uniq_id')

    def __init__(self, slope=1, uniq_id=0):
        self.slope = slope
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
        return hash(self.slope)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ','.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def c_headers(self):
        return ['<math.h>']

    def c_support_code(self):
        return mkl_helper.header_text() + """
            #define __DEBUG__ 0
            static int first_run = 1;
            static int count = 0;
            static int typenum;
            static int x_bs;
            static int x_channels;
            static int x_row;
            static int x_col;
            static dnnPrimitive_t batchNormBwdData;
            static dnnPrimitive_t batchNormBwdScaleShift;
            static size_t dim = 4;
            static size_t sizes[4];
            static size_t strides[4];
            static dnnError_t e;
            static dnnLayout_t bwd_bottom_diff_usr_l;
            static dnnLayout_t bwd_bottom_diff_int_l;
            static dnnPrimitive_t bwd_bottom_convert_to_int;
            static dnnPrimitive_t bwd_bottom_convert_from_int;
            static dnnPrimitive_t bwd_bottom_convert_prv2prv;
            static dnnLayout_t bwd_top_diff_usr_l;
            static dnnLayout_t bwd_top_diff_int_l;
            static dnnPrimitive_t bwd_top_convert_to_int;
            static dnnPrimitive_t bwd_top_convert_from_int;
            static dnnPrimitive_t bwd_top_convert_prv2prv;
            static dnnPrimitive_t* reluFwd_p  = static_cast<dnnPrimitive_t*>(NULL);
            static dnnPrimitive_t reluFwd  = static_cast<dnnPrimitive_t>(NULL);
            static dnnPrimitive_t reluBwd  = static_cast<dnnPrimitive_t>(NULL);
            static void* relu_res[dnnResourceNumber];
            static unsigned int long ip;
            static int* tmp = NULL;
            static void *output_buffer_ptr;
            static void *input_buffer_ptr;
            static void *input_gz;
        """

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
        else:
            raise TypeError('input must be float32 or float64')

        return """
        dnnReleaseBuffer_%(precision)s(output_buffer_ptr);
        """ % sub

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def make_node(self, x, gz):
        x = as_tensor_variable(x)
        gz = as_tensor_variable(gz)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        return Apply(self, [x, gz], [x.type()])

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        z, = out
        slope = self.slope

        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
        else:
            raise TypeError('input must be float32 or float64')

        sub = sub.copy()
        sub.update(locals())

        ret = """
        {
            #if __DEBUG__
            std::cout<<"Relu bwd start"<<std::endl;
            #endif
            if(first_run){
                x_bs = PyArray_DIMS(%(x)s)[0];
                x_channels = PyArray_DIMS(%(x)s)[1];
                x_row = PyArray_DIMS(%(x)s)[2];
                x_col = PyArray_DIMS(%(x)s)[3];
                //std::cout<<"bwd x shape "<<x_bs<<" channel "<<x_channels<<" row "<<x_row<<" col "<<x_col<<std::endl;
                sizes[0] = x_col;
                sizes[1] = x_row;
                sizes[2] = x_channels;
                sizes[3] = x_bs;
                strides[0] = 1;
                strides[1] = sizes[0];
                strides[2] = sizes[0]*sizes[1];
                strides[3] = sizes[0]*sizes[1]*sizes[2];
            }
            #if __DEBUG__
            printf(\"gz: %%d, %%d, %%d, %%d\\n\", PyArray_DIMS(%(gz)s)[0], PyArray_DIMS(%(gz)s)[1], PyArray_DIMS(%(gz)s)[2], PyArray_DIMS(%(gz)s)[3]);
            printf(\"x: %%d, %%d, %%d, %%d\\n\", PyArray_DIMS(%(x)s)[0], PyArray_DIMS(%(x)s)[1], PyArray_DIMS(%(x)s)[2], PyArray_DIMS(%(x)s)[3]);
            #endif

            if ( !(%(z)s
                    && PyArray_NDIM(%(z)s) == 4
                    && PyArray_IS_C_CONTIGUOUS(%(z)s)))
            {
                Py_XDECREF(%(z)s);
                %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(gz)s),
                                                  PyArray_DIMS(%(gz)s),
                                                  PyArray_TYPE(%(gz)s),
                                                  0);
                if (NULL == %(z)s) {
                    PyErr_Format(PyExc_RuntimeError,
                                "relu backward: Failed to allocate output");
                    %(fail)s
                }
            }

            // get internal layout for topgrad from previous Op
            bwd_bottom_diff_int_l = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];
            input_buffer_ptr = ((void**)PyArray_DATA(%(x)s))[1];

            // get internal buffer for topgrad from previous op
            bwd_top_diff_int_l = ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0];
            input_gz = ((void **)PyArray_DATA(%(gz)s))[1];
            //printf(\"reluGrad:%%x, %%x\\n\",bwd_top_diff_int_l,input_gz);

            if(first_run){
                if (E_SUCCESS != dnnReLUCreateBackward_%(precision)s(&reluBwd, NULL, bwd_bottom_diff_int_l,
                    bwd_bottom_diff_int_l, %(slope)s)){
                    std::cout<<"relu bwd creat fail\\n";
                }
            }
            if (NULL == output_buffer_ptr) {
                e = dnnAllocateBuffer_%(precision)s(&output_buffer_ptr, bwd_bottom_diff_int_l);
                if (E_SUCCESS != e){
                  std::cout<<"relu allocate fail with error code "<<e<<std::endl;
                }
                memset(output_buffer_ptr,0,dnnLayoutGetMemorySize_%(precision)s(bwd_bottom_diff_int_l));
            }

            if(dnnLayoutGetMemorySize_%(precision)s(bwd_bottom_diff_int_l) != dnnLayoutGetMemorySize_%(precision)s(bwd_top_diff_int_l))
            {
                printf(\"reluGradSize Error: %%d, %%d\\n\",
                       dnnLayoutGetMemorySize_%(precision)s(bwd_bottom_diff_int_l),
                       dnnLayoutGetMemorySize_%(precision)s(bwd_top_diff_int_l));
            }

            relu_res[dnnResourceSrc] = input_buffer_ptr;
            relu_res[dnnResourceDiffDst] = input_gz;
            relu_res[dnnResourceDiffSrc] = output_buffer_ptr;

            #if __DEBUG__
                std::cout<<"relu bwd, bwd_bottom_diff_int_l:"<<bwd_bottom_diff_int_l<<std::endl;
                std::cout<<"relu bwd, bwd_top_diff_int_l:"<<bwd_top_diff_int_l<<std::endl;
                std::cout<<"relu bwd, relu_res[dnnResourceSrc]:"<<relu_res[dnnResourceSrc]<<std::endl;
                std::cout<<"relu bwd, relu_res[dnnResourceDiffSrc]:"<<relu_res[dnnResourceDiffSrc]<<std::endl;
                std::cout<<"relu bwd, relu_res[dnnResourceDiffDst]:"<<relu_res[dnnResourceDiffDst]<<std::endl;
                std::cout<<"relu bwd, input size:"<<dnnLayoutGetMemorySize_%(precision)s(bwd_bottom_diff_int_l)<<std::endl;
                std::cout<<"relu bwd, output size:"<<dnnLayoutGetMemorySize_%(precision)s(bwd_top_diff_int_l)<<std::endl;
            #endif

            e = dnnExecute_%(precision)s(reluBwd, relu_res);
            if (E_SUCCESS != e){
                std::cout<<"relu bwd execute failed, e="<<e<<std::endl;
            }

            ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = bwd_bottom_diff_int_l;
            ((void**)PyArray_DATA(%(z)s))[1] = output_buffer_ptr;

            #if __DEBUG__
                printf(\"%%x, %%x\\n\",((dnnLayout_t*)PyArray_DATA(%(z)s))[0],((void**)PyArray_DATA(%(z)s))[1]);
                std::cout<<"relu bwd end\\n"<<std::endl;;
            #endif
            first_run = 0;
            }
        """ % sub
        return ret

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)

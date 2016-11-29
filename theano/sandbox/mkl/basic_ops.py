import theano.tensor as T
from theano.gof import Apply, Op
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import mkl_helper
from theano.gradient import DisconnectedType


class MKLOp(Op):
    def __init__(self, uniq_id=0):
        self.uniq_id = uniq_id

    def c_headers(self):
        return super(MKLOp, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(MKLOp, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        ccode = mkl_helper.header_text()
        ccode += """
            //#define _DEBUG_
            #define DIMENSION  4

            #define CHECK_ERR(f, err) \\
                do { \\
                    (err) = (f); \\
                    if ((err) != E_SUCCESS) { \\
                        printf("Error in file [%s:%d], err code (%d)", \\
                               __FILE__, __LINE__, err); \\
                        exit(1); \\
                    } \\
                } while(0)

            static dnnError_t err;
            static int first_run = 1;
            static void* internal_ptr = NULL; //mkl data buffer
            static void* usr_ptr = NULL;
            static dnnLayout_t layout_int = NULL;
            static dnnLayout_t layout_usr = NULL;
            static dnnPrimitive_t convert_to_int = NULL;
            static dnnPrimitive_t convert_from_int = NULL;
            static dnnPrimitive_t primitive = NULL;
            void *convert_resources[dnnResourceNumber];
            size_t bottomSize[DIMENSION] = {0};
            size_t bottomStride[DIMENSION] = {0};
        """
        return ccode

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)


class U2IPool(MKLOp):
    __props__ = ('ignore_border', 'mode', 'uniq_id')

    def __init__(self, ignore_border=False, mode='max', uniq_id=0):
        super(U2IPool, self).__init__(uniq_id)
        self.ignore_border = ignore_border
        self.mode = mode

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
        return hash(self.ignore_border) ^ hash(self.mode)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x, ws, stride=None, pad=(0, 0)):
        x = T.as_tensor_variable(x)
        if stride is None:
            stride = ws

        ws = T.as_tensor_variable(ws)
        stride = T.as_tensor_variable(stride)
        pad = T.as_tensor_variable(pad)

        broad = x.broadcastable[:2] + (False, False)
        out = T.TensorType(x.dtype, broad)
        return Apply(self, [x, ws, stride, pad], [out()])

    def grad(self, inp, grads):
        x, ws, stride, pad = inp
        gz, = grads
        disc = [DisconnectedType()() for i in inp[1:]]

        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)] + disc

    # def c_cleanup_code_struct(self, node, name):
    #    pass

    def c_code(self, node, name, inp, out, sub):
        x, ws, stride, pad = inp
        z, = out

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise Exception("Type %s is not supported!" %
                            node.inputs[0].type.dtype)

        fail = sub['fail']

        ignore_border = self.ignore_border
        if 'max' == self.mode:
            algo = "dnnAlgorithmPoolingMax"
        elif self.mode.startswith('average'):
            algo = "dnnAlgorithmPoolingAvg"
        else:
            raise NotImplementedError('%s is not implemented' % self.mode)

        ccode = """
            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3];  //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2];  //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1];  //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0];  //n
                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                size_t kernel_h = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
                size_t kernel_w = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
                size_t stride_h = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
                size_t stride_w = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
                size_t pad_h = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
                size_t pad_w = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));

                size_t kernelSize[2] = {kernel_w, kernel_h};
                size_t kernelStride[2] = {stride_w, stride_h};
                int inputOffset[2] = {pad_w, pad_h};

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, bottomSize, bottomStride), err );

                CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&primitive, NULL, %(algo)s,
                           layout_usr, kernelSize, kernelStride, inputOffset, dnnBorderZeros), err );

                //create internal layout
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_int, primitive, dnnResourceSrc), err );

                if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_int)) {
                    if(NULL == convert_to_int) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_to_int, layout_usr, layout_int), err );
                    }
                }
            }

            if (NULL == %(z)s) {
                //create PyArrayObject for output
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);
                if(NULL == %(z)s) {
                    %(fail)s
                }

                if (NULL == internal_ptr) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&internal_ptr, layout_int), err );
                }
            }

            if (convert_to_int) {
                convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
                convert_resources[dnnResourceTo] = (void *)(internal_ptr);

                CHECK_ERR( dnnExecute_%(precision)s(convert_to_int, convert_resources), err );
            } else {
                internal_ptr = (PyArray_DATA(%(x)s));
            }

            if(layout_int != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0])
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_int;
            if(internal_ptr != ((void**)PyArray_DATA(%(z)s))[1])
                ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr;

            first_run = 0;

            #ifdef __DEBUG__
                printf(\"U2IPool: from_buffer %%x to_buffer %%x\\n\",convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode

    def connection_pattern(self, node):
        return [[1], [0], [0], [0]]


class I2U(MKLOp):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        super(I2U, self).__init__(uniq_id)

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
        return hash(type(self))

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        out = x.type()
        return Apply(self, [x], [out])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [I2UGrad(uniq_id=self.uniq_id)(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
            x_item_size = 4
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
            x_item_size = 8
        else:
            raise Exception("Type %s is not supported!" %
                            node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            int status = 0;
            if (NULL == %(z)s) {
                %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);
                if (NULL == %(z)s) {
                    %(fail)s
                }

                int ndim = (int)PyArray_NDIM(%(x)s);
                size_t *bottom_size = (size_t *)malloc(ndim * sizeof(size_t));
                size_t *out_stride = (size_t *)malloc(ndim * sizeof(size_t));
                if(NULL == bottom_size || NULL == out_stride) {
                    printf(\"ERROR: malloc buffer in I2U \\n\");
                    exit(-1);
                }

                npy_intp dataSize = 1;
                for(int i = 0; i < ndim; i++) {
                    bottom_size[i] = (size_t)PyArray_DIMS(%(z)s)[ndim-i-1];
                    out_stride[i] = (size_t)PyArray_STRIDES(%(z)s)[ndim-i-1] / %(x_item_size)s;
                    dataSize = dataSize * bottom_size[i];
                }

                //create usr layerout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr,
                                                     ndim, bottom_size,
                                                     out_stride), err );

                free(bottom_size);
                free(out_stride);

                //Get layerout and internal buffer from input.
                layout_int = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];//get internal layerout
                internal_ptr = ((void**)PyArray_DATA(%(x)s))[1];

                CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_from_int, layout_int, layout_usr), err );
            }

            //FIXME, compare internal and user layout, then decides whether to do the conversion

            convert_resources[dnnResourceTo] = reinterpret_cast<void *>(PyArray_DATA(%(z)s));
            convert_resources[dnnResourceFrom] = reinterpret_cast<void *>(internal_ptr);

            //cvt
            CHECK_ERR( dnnExecute_%(precision)s(convert_from_int, convert_resources), err );
        """ % locals()

        return ccode


class U2IRelu(MKLOp):
    __props__ = ('slope', 'uniq_id')

    def __init__(self, slope=1, uniq_id=0):
        super(U2IRelu, self).__init__(uniq_id)
        self.slope = slope

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
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x):
        x = T.as_tensor_variable(x)

        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)]

    # def c_cleanup_code_struct(self, node, name):
    #    pass

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        slope = self.slope
        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise Exception("Type %s is not supported!" %
                            node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3];  //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2];  //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1];  //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0];  //n
                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, bottomSize, bottomStride), err );

                CHECK_ERR( dnnReLUCreateForward_%(precision)s(&primitive, NULL, layout_usr, %(slope)s), err );

                //create internal layout
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_int, primitive, dnnResourceSrc), err );

                if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_int)) {
                    if(NULL == convert_to_int) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_to_int, layout_usr, layout_int), err );
                    }
                }
            }

            if (NULL == %(z)s) {
                //create PyArrayObject for output
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);
                if(NULL == %(z)s) {
                    %(fail)s
                }

                if (NULL == internal_ptr) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&internal_ptr, layout_int), err );
                }
            }

            if (convert_to_int) {
                convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
                convert_resources[dnnResourceTo] = (void *)(internal_ptr);

                CHECK_ERR( dnnExecute_%(precision)s(convert_to_int, convert_resources), err );
            } else {
                internal_ptr = (PyArray_DATA(%(x)s));
            }

            if(layout_int != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0])
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_int;
            if(internal_ptr != ((void**)PyArray_DATA(%(z)s))[1])
                ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr;

            first_run = 0;

            #ifdef __DEBUG__
                printf(\"U2IRelu: from_buffer %%x to_buffer %%x\\n\",convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode


class U2IGrad(MKLOp):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        super(U2IGrad, self).__init__(uniq_id)

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
        return hash(type(self))

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x, gz):
        out = x.type()
        return Apply(self, [x, gz], [out])

    def c_cleanup_code_struct(self, node, name):
        d = {}
        if 'float32' == node.inputs[0].type.dtype:
            d['precision'] = "F32"
        elif "float64" == node.inputs[0].type.dtype:
            d['precision'] = "F64"
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)
        return """
              //free buffer
              int status = 0;

              if(NULL != usr_ptr)
              {
                  status = dnnReleaseBuffer_%(precision)s(usr_ptr);
                  if(0 != status)
                  {
                        printf(\"\\nERROR:dnnReleaseBuffer_%(precision)s usr_ptr in U2IGrad\\n\");
                        exit(0);
                  }

                  usr_ptr = NULL;
              }

              if(NULL != convert_from_int)
              {
                  status = dnnDelete_%(precision)s(convert_from_int);
                  if(0 != status)
                  {
                       printf(\"\\nERROR:dnnDelete_%(precision)s convert_to_int\\n\");
                  }
                  convert_from_int = NULL;
              }

              if(NULL != layout_usr)
              {
                  status = dnnLayoutDelete_%(precision)s(layout_usr);
                  if(0 != status)
                  {
                        printf(\"\\nERROR:dnnLayoutDelete__%(precision)s layout_usr in U2IGrad\\n\");
                        exit(0);
                  }
                  layout_usr = NULL;
              }
              """ % d

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        z, = out
        sub['x'] = x
        sub['gz'] = gz
        sub['z'] = z
        sub['name'] = U2IGrad.__name__
        if 'float32' == node.inputs[0].type.dtype:
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
        elif "float64" == node.inputs[0].type.dtype:
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)
        ccode = """
        int status = 0;
        if(NULL == %(z)s)
        {
              %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(gz)s),
                                          PyArray_DIMS(%(gz)s),
                                          PyArray_TYPE(%(gz)s),
                                          0);
              if(NULL == %(z)s)
              {
                    %(fail)s
              }

              int ndim = (int)PyArray_NDIM(%(gz)s);
              size_t*bottom_size = (size_t*)malloc(ndim*sizeof(size_t));
              size_t*out_stride = (size_t*)malloc(ndim*sizeof(size_t));
              if(0 == bottom_size || 0 == out_stride)
              {
                       printf(\"ERROR: malloc buffer in U2IGrad \\n\");
                       exit(0);
              }

              npy_intp dataSize = 1;
              for(int i=0;i<ndim;i++)
              {
                      bottom_size[i] = (size_t)PyArray_DIMS(%(z)s)[ndim-i-1];
                      out_stride[i] = (size_t)PyArray_STRIDES(%(z)s)[ndim-i-1] / %(x_item_size)s;
                      dataSize = dataSize * bottom_size[i];
              }

              //create usr layerout
              status = dnnLayoutCreate_%(precision)s(&layout_usr,
                                                     ndim, bottom_size,
                                                     out_stride);

              size_t size = dnnLayoutGetMemorySize_%(precision)s(layout_usr);
              if(size != PyArray_DIMS(%(z)s)[0]*PyArray_STRIDES(%(z)s)[0])
              {
                      printf(\"ERROR:dnnLayoutCreate_%(precision)s: %%d , %%d in U2IGrad\\n\",size, dataSize);
                      exit(0);
              }
              free(bottom_size);
              free(out_stride);

              //Get layerout buffer from input.
              layout_int = ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0]; //get internal layerout
              internal_ptr = ((void **)PyArray_DATA(%(gz)s))[1];

              status = dnnConversionCreate_%(precision)s(&convert_from_int, layout_int, layout_usr);
              if(0 != status)
              {
                     printf(\"ERROR:dnnConversionCreate_%(precision)s\\n\");
                     exit(0);
              }
        }

        convert_resources[dnnResourceFrom] = internal_ptr;
        convert_resources[dnnResourceTo] = (void*)PyArray_DATA(%(z)s);
        #ifdef _DEBUG_
            printf(\"%%x, %%x , %%x to %%x\\n\",convert_from_int,layout_int,internal_ptr,convert_resources[dnnResourceTo]);
        #endif
        if(dnnLayoutGetMemorySize_%(precision)s(layout_int) != PyArray_DIMS(%(z)s)[0]*PyArray_STRIDES(%(z)s)[0])
        {
               //printf(\"U2IGrad_Error: int: %%d != usr: %%d\\n\",dnnLayoutGetMemorySize_%(precision)s(layout_int),PyArray_DIMS(%(z)s)[0]*PyArray_STRIDES(%(z)s)[0]);
               //printf(\"U2IGrad DIM: %%d, %%d, %%d, %%d\\n\",PyArray_DIMS(%(z)s)[0],PyArray_DIMS(%(z)s)[1],PyArray_DIMS(%(z)s)[2],PyArray_DIMS(%(z)s)[3]);
        }
        //cvt
        status = dnnExecute_%(precision)s(convert_from_int, convert_resources);
        if(0 != status)
        {
                printf(\"ERROR:U2IGrad:%%x, %%x, %%x, status: %%d\\n\",convert_from_int,convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo],status);
                exit(0);
        }
        """ % sub
        return ccode


class I2UGrad(MKLOp):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        super(I2UGrad, self).__init__(uniq_id)

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
        return hash(type(self))

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x, gz):
        out = x.type()
        return Apply(self, [x, gz], [out])

    def c_cleanup_code_struct(self, node, name):
        d = {}
        d['name'] = I2UGrad.__name__
        if 'float32' == node.inputs[0].type.dtype:
            d['precision'] = "F32"
        elif "float64" == node.inputs[0].type.dtype:
            d['precision'] = "F64"
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)
        return """
              //free buffer
              int status = 0;
       #ifdef _DEBUG_
              printf(\"I2UGrad CleanUp\\n\");
       #endif
              if(NULL != convert_to_int)
              {
                    int status = dnnDelete_%(precision)s(convert_to_int);
                    if(0 != status)
                    {
                          printf(\"\\nERROR:dnnDelete_%(precision)s convert_to_int\\n\");
                    }
                    convert_to_int = NULL;
              }

              if(NULL != internal_ptr)
              {
                     status = dnnReleaseBuffer_%(precision)s(internal_ptr);
                     if(0 != status)
                     {
                             printf(\"\\nERROR:dnnReleaseBuffer_%(precision)s free internal buffer\\n\");
                      }
                      internal_ptr = NULL;
              }

              if(NULL != layout_usr)
              {
                       status = dnnLayoutDelete_%(precision)s(layout_usr);
                       if(0 != status)
                       {
                               printf(\"\\nERROR:dnnLayoutDelete__%(precision)s layout_usr in %(name)s\\n\");
                               exit(0);
                       }
                       layout_usr = NULL;
              }

       #ifdef _DEBUG_
              printf(\"I2UGrad CleanUp End\\n\");
       #endif
              """ % d

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        z, = out
        sub['x'] = x
        sub['z'] = z
        sub['gz'] = gz

        if 'float32' == node.inputs[0].type.dtype:
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
            sub['type'] = "float"
        elif "float64" == node.inputs[0].type.dtype:
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
            sub["type"] = "double"
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)

        ccode = """
      int status = 0;
      if(NULL == %(z)s)
      {
            npy_intp *dims = (npy_intp*)malloc(sizeof(npy_intp) * PyArray_NDIM(%(x)s));
            if(NULL == dims)
            {
                 printf(\"ERROR: malloc in I2UGrad\\n\");
                 exit(0);
            }

            %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),//I2UGrad
                                         PyArray_DIMS(%(x)s),//owen
                                         PyArray_TYPE(%(x)s),
                                         0);
             if(NULL == %(z)s)
             {
                 %(fail)s
             }
             free(dims);

             int ndim = (int)PyArray_NDIM(%(gz)s);
             size_t*bottom_size = (size_t*)malloc(ndim*sizeof(size_t));
             size_t*out_stride = (size_t*)malloc(ndim*sizeof(size_t));
             if(0 == bottom_size || 0 == out_stride)
             {
                    printf(\"ERROR: malloc buffer in I2U \\n\");
                    exit(0);
             }

             for(int i=0;i<ndim;i++)
             {
                    bottom_size[i] = (size_t)PyArray_DIMS(%(gz)s)[ndim-i-1];
                    out_stride[i] = (size_t)PyArray_STRIDES(%(gz)s)[ndim-i-1] / %(x_item_size)s;
             }

              //create usr layerout for gz
             status = dnnLayoutCreate_%(precision)s(&layout_usr,
                                                     ndim, bottom_size,
                                                     out_stride);
             if(0 != status)
             {
                      printf(\"ERROR:dnnLayoutCreate_%(precision)s\\n\");
                      exit(0);
             }
             free(bottom_size);
             free(out_stride);

             layout_int = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0]; //get internal layerout

             //create internal buffer for gradI
             if(NULL == internal_ptr)
             {
                       status = dnnAllocateBuffer_%(precision)s(
                                         reinterpret_cast<void **>(&internal_ptr),
                                         layout_int);
                       if(0 != status)
                       {
                               printf(\"ERROR:dnnAllocateBuffer_%(precision)s : %%d \\n\", status);
                               exit(0);
                       }
              }

              if(dnnLayoutGetMemorySize_%(precision)s(layout_usr) != dnnLayoutGetMemorySize_%(precision)s(layout_int))
              {
                            printf(\"ERROR:I2UGrad: usr space: %%d not equal to internal:%%d\\n\",
                                            dnnLayoutGetMemorySize_%(precision)s(layout_usr), dnnLayoutGetMemorySize_%(precision)s(layout_int));
                            exit(0);
              }

              status = dnnConversionCreate_%(precision)s(&convert_to_int, layout_usr, layout_int);
              if(0 != status)
              {
                    printf(\"ERROR: dnnConversionCreate_%(precision)s in I2UGrad\\n\");
                    exit(0);
              }

              ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr;
              ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_int;
      }

      convert_resources[dnnResourceTo] = reinterpret_cast<void *>(internal_ptr);
      convert_resources[dnnResourceFrom] = reinterpret_cast<void *>(PyArray_DATA(%(gz)s));

  #ifdef _DEBUG_
      printf(\"I2UGrad:%%x, %%x, %%x to %%x\\n\",convert_to_int,layout_int,convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
      printf(\"I2UGrad x: %%d,%%d,%%d,%%d\\n\",PyArray_DIMS(%(x)s)[0],PyArray_DIMS(%(x)s)[1],PyArray_DIMS(%(x)s)[2],PyArray_DIMS(%(x)s)[3]);
      printf(\"I2UGrad gz: %%d,%%d,%%d,%%d\\n\",PyArray_DIMS(%(gz)s)[0],PyArray_DIMS(%(gz)s)[1],PyArray_DIMS(%(gz)s)[2],PyArray_DIMS(%(gz)s)[3]);
  #endif

      if(dnnLayoutGetMemorySize_%(precision)s(layout_int) != PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0])
      {
            printf(\"I2UGrad int %%d != usr: %%d\\n\",dnnLayoutGetMemorySize_%(precision)s(layout_int),PyArray_DIMS(%(z)s)[0]*PyArray_STRIDES(%(z)s)[0]);
            printf(\"I2UGrad gz: %%d,%%d,%%d,%%d\\n\",PyArray_DIMS(%(gz)s)[0],PyArray_DIMS(%(gz)s)[1],PyArray_DIMS(%(gz)s)[2],PyArray_DIMS(%(gz)s)[3]);
            printf(\"I2UGrad x: %%d,%%d,%%d,%%d\\n\",PyArray_DIMS(%(x)s)[0],PyArray_DIMS(%(x)s)[1],PyArray_DIMS(%(x)s)[2],PyArray_DIMS(%(x)s)[3]);
      }

      //cvt
      status = dnnExecute_%(precision)s(convert_to_int, convert_resources);
      if(0 != status)
      {
                printf(\"ERROR:dnnExecute_%(precision)s\\n\");
                exit(0);
      }
     """ % sub
        return ccode


class U2ILRN(MKLOp):
    __props__ = ('slope', 'alpha', 'beta', 'k', 'size', 'uniq_id')

    def __init__(self, slope=1, alpha=1e-4, beta=0.75, k=2, n=5, uniq_id=0):
        super(U2ILRN, self).__init__(uniq_id)
        self.slope = slope
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.size = n

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
        return hash(self.slope) ^ hash(self.alpha) ^ hash(self.beta) ^ hash(self.k) ^ hash(self.size)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        slope = self.slope
        alpha = self.alpha
        beta = self.beta
        k = self.k
        size = self.size

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise Exception('Type %s is not supported!' % node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3]; //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2]; //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1]; //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0]; //n

                bottomStride[0] = 1;
                bottomStride[1] = bottomStride[0] * bottomSize[0];
                bottomStride[2] = bottomStride[1] * bottomSize[1];
                bottomStride[3] = bottomStride[2] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, bottomSize, bottomStride), err );
                CHECK_ERR( dnnLRNCreateForward_%(precision)s(&primitive, NULL, layout_usr, %(size)s, %(alpha)s, %(beta)s, %(k)s), err );
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_int, primitive, dnnResourceSrc), err );

                if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_int))
                {
                    if (NULL == convert_to_int)
                    {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_to_int, layout_usr, layout_int), err );
                    }
                }
            }

            if (NULL == %(z)s)
            {
                //Create PyArrayObject for output
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION, PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s), 0);

                if (NULL == %(z)s)
                {
                    %(fail)s
                }

                if (NULL == internal_ptr)
                {
                    CHECK_ERR(  dnnAllocateBuffer_%(precision)s((void**)&internal_ptr, layout_int), err );
                }
            }

            if (convert_to_int)
            {
                convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
                convert_resources[dnnResourceTo] = (void*)(internal_ptr);
                CHECK_ERR( dnnExecute_%(precision)s(convert_to_int, convert_resources), err );
            }
            else
            {
                internal_ptr = (PyArray_DATA(%(x)s));
            }
            if (layout_int != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0])
            {
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_int;
            }
            if (internal_ptr != ((void**)PyArray_DATA(%(z)s))[1])
            {
                ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr;
            }

            first_run = 0;

            #ifdef __DEBUG__
                printf(\"U2ILRN: from_buffer %%x to_buffer %%x\\n\", convert_resources[dnnResouceFrom], convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode

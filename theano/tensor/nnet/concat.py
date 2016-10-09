from theano.tensor import as_tensor_variable, TensorType
import theano
from theano.gof import Apply,HideC
from theano import gof
from theano.tensor.basic import Join, Split

class JoinMKL(HideC,Join):
    """
    Join for MKL
    """
    def __init__(self,axis):
       self.axis = axis
       super(JoinMKL, self).__init__()

    def __eq__(self,other):
       return (type(self) == type(other) and
                (self.axis == other.axis))

    def make_node(self, axis,*tensors):
       node = Join.make_node(self,axis,*tensors)
       def agv(v):
           return as_tensor_variable(v)
       return Apply(self,[as_tensor_variable(axis)] + list(map(agv, tensors)),
                                    [TensorType(dtype=node.outputs[0].dtype, 
                                    broadcastable=node.outputs[0].broadcastable)()])

    def __hash__(self):
       return hash(type(self)) ^ hash(self.axis)

    def __str__(self):
       return "MklJoin"

    def c_code_cache_version(self):
        return (1,0)

    def c_headers(self):
        return ['mkl.h']

    def c_libraries(self):
        return ['mkl_rt']

    def c_compile_args(self):
        return ['-O3']

    def c_support_code(self):
        final_code = """
                    """
        return final_code 

    def c_cleanup_code_struct(self,node, name):
        return """ """ 

    def c_init_code_struct(self, node, name, sub):
        return """ """

    def c_code(self, node, name, inputs, out, sub):
        tmp,tensors, = inputs[0],inputs[1:]
        z, = out
        input_l = tensors[0]
        L = len(tensors)
        sub['axis'] = self.axis
        sub['z'] = z
        sub["L"] = L
        sub["ndim"] = node.inputs[0].type.ndim
        print (node.inputs[0].type.ndim)
        print (node.inputs[0].type.dtype)
        if 'float32' == node.inputs[1].type.dtype:
            sub['type'] = "float"
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
        elif "float64" == node.inputs[1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
        else:
            raise  Exception("Type %s not implemented" %
                                node.inputs[1].type.dtype)
        sub['x'] = tensors[0]
        ccode = """  
               dnnLayout_t layout_int[%(L)s];
               dnnPrimitive_t pConcat = NULL;
               void* internal_ptr = NULL;
               dnnLayout_t out_layer = NULL;
               void *concat_res[dnnResourceNumber];
               int status = 0;
               npy_intp out_dim[%(ndim)s];

               memcpy(out_dim, PyArray_DIMS(%(x)s), PyArray_NDIM(%(x)s)*sizeof(npy_intp));
               out_dim[%(axis)s] = 0;

               if(NULL == pConcat)
               {
               """ % sub
        for i, inp in enumerate(tensors):
            d={}
            d['i'] = i
            d['inp'] = inp
            d['axis'] = self.axis
            ccode += """
                  layout_int[%(i)s] = ((dnnLayout_t*)PyArray_DATA(%(inp)s))[0];
                  concat_res[dnnResourceMultipleSrc + %(i)s] = ((void**)PyArray_DATA(%(inp)s))[1];
                  out_dim[%(axis)s] = out_dim[%(axis)s] + PyArray_DIMS(%(inp)s)[%(axis)s];
                  """ % d
        ccode += """
                    status = dnnConcatCreate_%(precision)s(&pConcat,NULL,%(L)s,layout_int);
                    if(0 != status)
                    {
                         printf(\"ERROR:Create %(L)s primitive for concat\\n\");
                         exit(0);
                    }
               }

               //create PyArrayObject
               %(z)s = (PyArrayObject*)PyArray_ZEROS(%(ndim)s,
                                          out_dim,
                                          PyArray_TYPE(%(x)s),
                                          0);
               if(NULL == %(z)s)
               {
                    %(fail)s
               }


               if(NULL == out_layer)
               {
                    status = dnnLayoutCreateFromPrimitive_%(precision)s(&out_layer,pConcat,dnnResourceDst);
                    if(0 != status)
                    {
                         printf(\"ERROR:create Primitive layerout\\n\");
                         exit(0);
                    }
               }

               if(NULL == internal_ptr)
               {
                    status = dnnAllocateBuffer_%(precision)s(
                                   reinterpret_cast<void **>(&internal_ptr),
                                   out_layer);
               #ifdef _DEBUG_
                   printf(\"Create InternalBuffer: %%x\\n\",internal_ptr);
               #endif
               }
               concat_res[dnnResourceDst] = internal_ptr;
               status = dnnExecute_%(precision)s(pConcat, concat_res);
               if(0 != status)
               {
                   printf(\"ERROR:Concat Execute\\n\"); 
                   exit(0);
               }
                     
               ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = out_layer;
               ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr;
               """ % sub
        return ccode 
        

class concatGrad(gof.Op):
    def __init__(self, Op, target, params):
       self.params = params
       self.Op = Op
       self.target = target

    def __hash__(self):
       return hash(type(self)) ^ hash(self.params) ^ hash(self.Op) ^ hash(self.target)

    def __eq__(self,other):
       return (type(self) == type(other) and
                self.Op == other.Op and
                self.params == other.params and
                self.target == other.target)

    def __str__(self):
       return "U2IGrad_%s"%(self.Op)

    def c_code_cache_version(self):
        return (1,0)

    def c_headers(self):
        return ['mkl.h']

    def c_libraries(self):
        return ['mkl_rt']

    def c_compile_args(self):
        return ['-O3']

    def make_node(self, x, gz):
        out = x.type()
        return gof.Apply(self,[x, gz],[out])

    def c_support_code(self):
        return """
        // #define _DEBUG_
         static void* internal_ptr = NULL;
         static void* usr_ptr = NULL;
         static dnnLayout_t layout_int = NULL;
         static dnnLayout_t layout_usr = NULL;
         static dnnPrimitive_t convert_from_int = NULL;
         static dnnPrimitive_t FilterFwd = NULL;
         void *convert_resources[dnnResourceNumber];
         """

    def c_cleanup_code_struct(self,node, name):
        d={}
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
        sub['target'] = self.target
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
         }
  #if 1
         if(NULL == layout_usr)
         {
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
	}

        //Get layerout buffer from input.
        layout_int = ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0]; //get internal layerout
        internal_ptr = ((void **)PyArray_DATA(%(gz)s))[1];
        if(PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0] != dnnLayoutGetMemorySize_%(precision)s(layout_int))
        {
               printf(\"%(name)s ERROR: User space: %%d not equal to internal: %%d\\n\",
                               PyArray_DIMS(%(gz)s)[0] * PyArray_STRIDES(%(gz)s)[0],dnnLayoutGetMemorySize_%(precision)s(layout_int));
                    exit(0);
        }


        if(NULL == convert_from_int)
        {
               status = dnnConversionCreate_%(precision)s(&convert_from_int, layout_int, layout_usr);
               if(0 != status)
               {
                     printf(\"ERROR:dnnConversionCreate_%(precision)s, U2IGrad\\n\");
                     exit(0);
               }
        }

        if(internal_ptr != convert_resources[dnnResourceFrom])
        {
               convert_resources[dnnResourceFrom] = internal_ptr;
        }
       
        if(PyArray_DATA(%(z)s) != convert_resources[dnnResourceTo])
        {
               convert_resources[dnnResourceTo] = (void*)PyArray_DATA(%(z)s);
        }

        //cvt
        status = dnnExecute_%(precision)s(convert_from_int, convert_resources);
        if(0 != status)
        {
                printf(\"ERROR:U2IGrad:%%x, %%x, %%x, status: %%d\\n\",convert_from_int,convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo],status);
                exit(0);
        }
   #endif
        """ % sub
        return ccode


class I2U(gof.Op):
    def __init__(self, uniq_name=None):
        self.uniq_name = uniq_name

    def __hash__(self):
       return (hash(type(self)) ^ hash(self.uniq_name))

    def __eq__(self,other):
       return (type(self) == type(other) \
               and self.uniq_name == other.uniq_name)

    def __str__(self):
       return "I2U_%s"%(self.uniq_name)

    def c_code_cache_version(self):
        return (1, 0, hash(self.uniq_name))

    def c_headers(self):
        return ['mkl.h']

    def c_libraries(self):
        return ['mkl_rt']

    def c_compile_args(self):
        return ['-O3']

    def make_node(self, x):
        out = x.type()
        return gof.Apply(self,[x],[out])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [I2UGrad(self.uniq_name)(x, gz)]

    def c_support_code(self):
        return """ 
         //#define _DEBUG_
         static void* internal_ptr = NULL;
         static dnnLayout_t layout_int = NULL;
         static dnnLayout_t layout_usr = NULL;
         static dnnPrimitive_t convert_from_int = NULL;
         void *convert_resources[dnnResourceNumber];
         """

    def c_cleanup_code_struct(self,node, name):
        d={}
        if 'float32' == node.inputs[0].type.dtype:
            d['precision'] = "F32"
        elif "float64" == node.inputs[0].type.dtype:
            d['precision'] = "F64"
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)
        return """
              //free buffer
              dnnError_t status;
              if(NULL != convert_from_int)
              {
                    status = dnnDelete_%(precision)s(convert_from_int);
                    if(0 != status)
                    {
                            printf(\"\\nERROR:dnnDelete_%(precision)s convert_from_int\\n\");
                            exit(0);
                    }
                    convert_from_int = NULL;
              }

              if(NULL != layout_usr)
              {
                    status = dnnLayoutDelete_%(precision)s(layout_usr);
                    if(0 != status)
                    {
                          printf(\"\\nERROR:dnnLayoutDelete__%(precision)s layout_usr in\\n\");
                          exit(0);
                    }
                    layout_usr = NULL;
              }
              """ % d

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        sub['x'] = x
        sub['z'] = z
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
                  %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),
                                          PyArray_DIMS(%(x)s),
                                          PyArray_TYPE(%(x)s),
                                          0);
                  if(NULL == %(z)s)
                  {
                      %(fail)s
                  }
               }
    #if 1 
         #ifdef _DEBUG_
               printf(\"x: %%d, %%d, %%d, %%d\\n\",PyArray_DIMS(%(x)s)[0],PyArray_DIMS(%(x)s)[1],PyArray_DIMS(%(x)s)[2],PyArray_DIMS(%(x)s)[3]);
         #endif
               if(NULL == layout_usr)
               {   
                  int ndim = (int)PyArray_NDIM(%(x)s);
                  size_t*bottom_size = (size_t*)malloc(ndim*sizeof(size_t));
                  size_t*out_stride = (size_t*)malloc(ndim*sizeof(size_t));
                  if(0 == bottom_size || 0 == out_stride)
                  {
                       printf(\"ERROR: malloc buffer in I2U \\n\");
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
                  if(size != dataSize*%(x_item_size)s)
                  {
                            printf(\"ERROR:I2U dnnLayoutCreate_%(precision)s: %%d , %%d\\n\",size, dataSize);
                            exit(0);
                  }
                  free(bottom_size);
                  free(out_stride);
               }//out

               //Get layerout and internal buffer from input.
               layout_int = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];//get internal layerout
               internal_ptr = ((void**)PyArray_DATA(%(x)s))[1];
               if(PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0] != dnnLayoutGetMemorySize_%(precision)s(layout_int))
               {
                         printf(\"usr size != internal size : %%d, %%d\\n\",PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0],
                                                    dnnLayoutGetMemorySize_%(precision)s(layout_int));
                         exit(0);
               }


               if(PyArray_DATA(%(z)s) != convert_resources[dnnResourceTo])
               {
                       convert_resources[dnnResourceTo] = reinterpret_cast<void *>(PyArray_DATA(%(z)s));
               }

               if(internal_ptr != convert_resources[dnnResourceFrom])
               {
                       convert_resources[dnnResourceFrom] = reinterpret_cast<void *>(internal_ptr);
               }
               if(NULL == convert_from_int)
               {
                       status = dnnConversionCreate_%(precision)s(&convert_from_int, layout_int, layout_usr);
               }
               if(status !=0)
               {
		       printf(\"I2U error: %%d\\n\",status);
                       memcpy(convert_resources[dnnResourceTo],convert_resources[dnnResourceFrom],dnnLayoutGetMemorySize_%(precision)s(layout_int));
               }
               else
               {
          #ifdef _DEBUG_
                       printf(\"I2UInt:%%x\\n\",convert_from_int);
                       printf(\"I2U %%x to %%x\\n\",convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
          #endif
                       status = dnnExecute_%(precision)s(convert_from_int, convert_resources);
                       if(0 != status)
                       {
                              printf(\"ERROR: convert_from_int in I2U\\n\");
                              exit(0);
                       }
               }
           #ifdef _DEBUG_
               printf(\"I2U %%x to %%x\\n\",convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
               printf(\"I2U %%f, %%f\\n\",((float*)PyArray_DATA(%(z)s))[1],((float*)PyArray_DATA(%(z)s))[3]);
           #endif
         #endif
               """ % sub
        return ccode         



class I2UGrad(gof.Op):
    def __init__(self, uniq_name):
        self.uniq_name = uniq_name

    def __hash__(self):
       return (hash(type(self)) ^ hash(self.uniq_name))

    def __eq__(self,other):
       return (type(self) == type(other) \
               and self.uniq_name == other.uniq_name)

    def __str__(self):
       return "I2UGrad"

    def c_code_cache_version(self):
        return (1,0, hash(self.uniq_name))

    def c_headers(self):
        return ['mkl.h']

    def c_libraries(self):
        return ['mkl_rt']

    def c_compile_args(self):
        return ['-O3']

    def make_node(self,x,gz):
        out = x.type()
        return gof.Apply(self,[x, gz],[out])

    def c_support_code(self):
        return """
         //#define _DEBUG_
         static void* internal_ptr = NULL;
         static dnnLayout_t layout_int = NULL;
         static dnnLayout_t layout_usr = NULL;
         static dnnPrimitive_t convert_to_int = NULL;
         void *convert_resources[dnnResourceNumber];
         """

    def c_cleanup_code_struct(self,node, name):
        d={}
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
            %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),//I2UGrad
                                         PyArray_DIMS(%(x)s),//owen
                                         PyArray_TYPE(%(x)s),
                                         0);
             if(NULL == %(z)s)
             {
                 %(fail)s
             }
        }
 #if 1 
        if(NULL == internal_ptr)
        {
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
                               printf(\"I2UGrad ERROR:dnnAllocateBuffer_%(precision)s : %%d \\n\", status);
                               exit(0);
                       }
               }

               if(PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0] != dnnLayoutGetMemorySize_%(precision)s(layout_int))
               {
                            printf(\"ERROR:I2UGrad: usr space: %%d not equal to internal:%%d\\n\",
                                            PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0], dnnLayoutGetMemorySize_%(precision)s(layout_int));
                            exit(0);
               }
      }

      //record mkl_DNN info 
      if(internal_ptr != ((void**)PyArray_DATA(%(z)s))[1])
                ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr;//record internal buff

      if(layout_int != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0])
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_int; //layout int


      if(NULL == convert_to_int)
      {
               status = dnnConversionCreate_%(precision)s(&convert_to_int, layout_usr, layout_int);
               if(0 != status)
               {
                    printf(\"ERROR: dnnConversionCreate_%(precision)s in I2UGrad\\n\");
                    exit(0);
               }
      }

      if(internal_ptr != convert_resources[dnnResourceTo])
      {
               convert_resources[dnnResourceTo] = reinterpret_cast<void *>(internal_ptr);

      }

      if(PyArray_DATA(%(gz)s) != convert_resources[dnnResourceFrom])
      {
               convert_resources[dnnResourceFrom] = reinterpret_cast<void *>(PyArray_DATA(%(gz)s));
      }

       //cvt
       status = dnnExecute_%(precision)s(convert_to_int, convert_resources);
       if(0 != status)
       {
                printf(\"ERROR:dnnExecute_%(precision)s\\n\");
                exit(0);
       }
  #ifdef _DEBUG_
       printf(\"I2UGradSize:%%d, %%d\\n\",dnnLayoutGetMemorySize_%(precision)s(layout_int), dnnLayoutGetMemorySize_%(precision)s(layout_usr));
       printf(\"I2UGrad: %%x, %%x\\n\",convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
  #endif
 #endif
     """ % sub
        return ccode    

 

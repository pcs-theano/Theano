from theano.tensor import as_tensor_variable, TensorType
import theano
from theano.gof import Apply,HideC
from theano import gof
from theano.tensor.basic import Join, Split

class JoinMKL3(gof.Op):
    """
    Join for MKL
    """
    def __init__(self,axis,uniq_id=1):
       self.axis = axis
       self.uniq_id=uniq_id

    def __eq__(self,other):
       return (type(self) == type(other) and
                (self.axis == other.axis) and
                (self.uniq_id == other.uniq_id))

    def make_node(self, x,y,z):
       return Apply(self,[x,y,z],[x.type()])

    def __hash__(self):
       return hash(type(self)) ^ hash(self.axis) ^ hash(self.uniq_id)

    def grad(self, inp, grads):
       x, y, z, = inp
       gz, = grads
       gx, gy, gm = JoinMKLGrad3(self.axis,self.uniq_id)(x, y, z, gz)
       return gx, gy, gm  

    def __str__(self):
       return "MklJoin"

    def c_code_cache_version(self):
        return (1,0,hash(self.uniq_id))

    def c_headers(self):
        return ['mkl.h']

    def c_libraries(self):
        return ['mkl_rt']

    def c_compile_args(self):
        return ['-O3']

    def c_support_code(self):
        final_code = """
               //#define _DEBUG_
               dnnPrimitive_t pConcat = NULL;
               void* internal_ptr = NULL;
               dnnLayout_t out_layer = NULL;
               void *concat_res[dnnResourceNumber];
                    """ 
        return final_code 

    def c_cleanup_code_struct(self,node, name):
        sub = {}
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
        return """
             int status = 0;
             if(NULL != pConcat)
             {
                  status = dnnDelete_%(precision)s(pConcat);
                  if(0 != status)
                  {
                        printf(\"ERROR:free pConcat\\n\");
                        exit(0);
                  }
                  pConcat = NULL;
             }

             if(NULL != internal_ptr)
             {
                  status = dnnReleaseBuffer_%(precision)s(internal_ptr);
                  if(0 != status)
                  {
                        printf(\"ERROR:free buffer  in JoinMKL\\n\");
                        exit(0);
                  }
                  internal_ptr = NULL;
             }

        #if 0
             if(NULL != out_layer)
             {
                  status = dnnLayoutDelete_%(precision)s(out_layer);
                  if(0 != status)
                  {
                        printf(\"ERROR:free out_layer in JoinMKL\\n\");
                        exit(0);
                  }
                  out_layer = NULL;
             }
         #endif
             """ % sub 

    def c_code(self, node, name, inputs, out, sub):
        x, y, k, = inputs
        z, = out
        sub['axis'] = self.axis
        sub['z'] = z
        sub["L"] = 3 
        sub["ndim"] = node.inputs[1].type.ndim
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
        sub['x'] = x
        ccode = """
               npy_intp out_dim[%(ndim)s];

               int status = 0;
               if(NULL == pConcat)
               {
                  dnnLayout_t layout_int[%(L)s];
                  memcpy(out_dim, PyArray_DIMS(%(x)s), PyArray_NDIM(%(x)s)*sizeof(npy_intp));
                  out_dim[%(axis)s] = 0;
               """ % sub
        for i, inp in enumerate([x,y,k]):
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
            """ % sub
        ccode += """
               if(NULL == %(z)s)
               {
                     //create PyArrayObject
                     %(z)s = (PyArrayObject*)PyArray_ZEROS(%(ndim)s,
                                          out_dim,
                                          PyArray_TYPE(%(x)s),
                                          0);
                      if(NULL == %(z)s)
                      {
                            %(fail)s
                      }
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

class JoinMKLGrad3(gof.Op):
    """
    Join for MKL
    """
    def __init__(self,axis,uniq_id=1):
       self.axis = axis
       self.uniq_id=uniq_id

    def __eq__(self,other):
       return (type(self) == type(other) and
                (self.axis == other.axis) and
                (self.uniq_id == other.uniq_id))

    def make_node(self, x,y, z,gz):
       return Apply(self,[x, y, z, gz],[x.type(),y.type(),z.type()])

    def __hash__(self):
       return hash(type(self)) ^ hash(self.axis) ^ hash(self.uniq_id)

    def __str__(self):
       return "MklJoinGrad_%d"%(self.uniq_id)

    def c_code_cache_version(self):
        return (1,0,hash(self.uniq_id))

    def c_headers(self):
        return ['mkl.h']

    def c_libraries(self):
        return ['mkl_rt']

    def c_compile_args(self):
        return ['-O3']

    def c_support_code(self):
        final_code = """
              // #define _DEBUG_
               dnnPrimitive_t pSplit = NULL;
               void *concat_res[dnnResourceNumber];
               void*bufferList[3] = {NULL};
               """ 
        return final_code 

    def c_cleanup_code_struct(self,node, name):
        sub = {}
        sub['L'] = len(node.inputs)
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
        return """
             int status = 0;
             if(NULL != pSplit)
             {
                 status = dnnDelete_%(precision)s(pSplit);
                 if(0 != status)
                 {
                      printf(\"ERROR:free pSplit\\n\");
                      exit(0);
                 }
                 pSplit = NULL;
             }
             for(int i = 0; i< %(L)s;i++)
             {
                 if(NULL != bufferList[i])
                 {
                       status = dnnReleaseBuffer_%(precision)s(bufferList[i]);
                       if(0 != status)
                       {
                             printf(\"ERROR: free buffer in JoinGrad\\n\");
                             exit(0);
                       }
                       bufferList[i] = NULL;
                 }
             }
             """ % sub 

    def c_code(self, node, name, inputs, out, sub):
        x, y, m, gz, = inputs
        gx, gy, gm, = out
        sub['axis'] = self.axis
        sub['gx'] = gx
        sub['gy'] = gy
        sub['gz'] = gz
        sub['gm'] = gm
        sub["L"] = 3
        sub["ndim"] = node.inputs[1].type.ndim
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
        sub['x'] = x
        sub['y'] = y
        sub['m'] = m
        ccode = """
               int status = 0;
               if(NULL == pSplit)
               {
                     size_t dstChannelSize[%(L)s] = {0};
                     dstChannelSize[0] = (size_t)PyArray_DIMS(%(x)s)[1];
                     dstChannelSize[1] = (size_t)PyArray_DIMS(%(y)s)[1];
                     dstChannelSize[2] = (size_t)PyArray_DIMS(%(m)s)[1];

                     status = dnnSplitCreate_%(precision)s(&pSplit, NULL, %(L)s,
                                        ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0],
                                         dstChannelSize);
                     if(0 != status)
                     {
                           printf(\"ERROR: dnnSplitCreate Primitive: %(L)s, layer:%%x, %%d, %%d, %%d\\n\",
                                        ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0],dstChannelSize[0],dstChannelSize[1],dstChannelSize[2]);
                           exit(0);
                     }
                }
               """ % sub
        ccode += """
                if(NULL == %(gx)s)
                {
                    %(gx)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),
                                                           PyArray_DIMS(%(x)s),
                                                           PyArray_TYPE(%(x)s),
                                                           0);
                      if(NULL == %(gx)s)
                      {
                            %(fail)s
                      }
                      dnnLayout_t layout_int = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];
                      ((dnnLayout_t*)PyArray_DATA(%(gx)s))[0] = layout_int;
                      void* internal_ptr = NULL;
              	      //create internal buffer
                      if(NULL == internal_ptr)
                      {
                               status = dnnAllocateBuffer_%(precision)s(
                                               reinterpret_cast<void **>(&internal_ptr),
                                               layout_int);
                           #ifdef _DEBUG_
                               printf(\"Create InternalBuffer: %%x\\n\",internal_ptr);
                           #endif
                      }
                      int size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_int);
                      if(size != PyArray_DIMS(%(gx)s)[0]*PyArray_STRIDES(%(gx)s)[0])
                      {
                                printf(\"ERROR:int buffer Size: %%d != usr: %%d\\n\",size,
                                                PyArray_DIMS(%(gx)s)[0]*PyArray_STRIDES(%(gx)s)[0]);
                                exit(0);            
                      }
                      memset(internal_ptr,0,size);
                      concat_res[dnnResourceMultipleDst + 0] = reinterpret_cast<void*>(internal_ptr);
                      bufferList[0] = internal_ptr;
                      ((void**)PyArray_DATA(%(gx)s))[1] = internal_ptr; 
                }
               """ % sub
        ccode += """
                if(NULL == %(gy)s)
                {
                    %(gy)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(y)s),
                                                           PyArray_DIMS(%(y)s),
                                                           PyArray_TYPE(%(y)s),
                                                           0);
                      if(NULL == %(gy)s)
                      {
                            %(fail)s
                      }
                      dnnLayout_t layout_int = ((dnnLayout_t*)PyArray_DATA(%(y)s))[0];
                      ((dnnLayout_t*)PyArray_DATA(%(gy)s))[0] = layout_int;
                      void* internal_ptr = NULL;
                      //create internal buffer
                      if(NULL == internal_ptr)
                      {
                               status = dnnAllocateBuffer_%(precision)s(
                                               reinterpret_cast<void **>(&internal_ptr),
                                               layout_int);
                           #ifdef _DEBUG_
                               printf(\"Create InternalBuffer: %%x\\n\",internal_ptr);
                           #endif
                      }
                      int size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_int);
                      if(size != PyArray_DIMS(%(gy)s)[0]*PyArray_STRIDES(%(gy)s)[0])
                      {
                                printf(\"ERROR:int buffer Size: %%d != usr: %%d\\n\",size,
                                                PyArray_DIMS(%(gy)s)[0]*PyArray_STRIDES(%(gy)s)[0]);
                                exit(0);
                      }
                      memset(internal_ptr,0,size);
                      concat_res[dnnResourceMultipleDst + 1] = reinterpret_cast<void*>(internal_ptr);
                      bufferList[1] = internal_ptr;
                      ((void**)PyArray_DATA(%(gy)s))[1] = internal_ptr;
                }
               """ % sub
        ccode += """
                if(NULL == %(gm)s)
                {
                     %(gm)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(m)s),
                                                           PyArray_DIMS(%(m)s),
                                                           PyArray_TYPE(%(m)s),
                                                           0);
                      if(NULL == %(gm)s)
                      {
                            %(fail)s
                      }
                      dnnLayout_t layout_int = ((dnnLayout_t*)PyArray_DATA(%(m)s))[0];
                      ((dnnLayout_t*)PyArray_DATA(%(gm)s))[0] = layout_int;
                      void* internal_ptr = NULL;
                      //create internal buffer
                      if(NULL == internal_ptr)
                      {
                               status = dnnAllocateBuffer_%(precision)s(
                                               reinterpret_cast<void **>(&internal_ptr),
                                               layout_int);
                           #ifdef _DEBUG_
                               printf(\"Create InternalBuffer: %%x\\n\",internal_ptr);
                           #endif
                      }
                      int size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_int);
                      if(size != PyArray_DIMS(%(gm)s)[0]*PyArray_STRIDES(%(gm)s)[0])
                      {
                                printf(\"ERROR:int buffer Size: %%d != usr: %%d\\n\",size,
                                                PyArray_DIMS(%(gm)s)[0]*PyArray_STRIDES(%(gm)s)[0]);
                                exit(0);
                      }
                      memset(internal_ptr,0,size);
                      concat_res[dnnResourceMultipleDst + 2] = reinterpret_cast<void*>(internal_ptr);
                      bufferList[2] = internal_ptr;
                      ((void**)PyArray_DATA(%(gm)s))[1] = internal_ptr;
                }
               """ % sub
        ccode += """
               concat_res[dnnResourceSrc] = reinterpret_cast<void*>(((void**)PyArray_DATA(%(gz)s))[1]);
        #ifdef _DEBUG_
               printf(\"split: %%x, %%x, %%x, %%x, %%x\\n\",pSplit,concat_res[dnnResourceSrc],
                                        concat_res[dnnResourceMultipleDst + 2],
                                        concat_res[dnnResourceMultipleDst + 1],
                                        concat_res[dnnResourceMultipleDst + 0]);
               printf(\"%%x, %%x, %%x,%%x\\n\",%(gx)s,((dnnLayout_t*)PyArray_DATA(%(gx)s))[0],
                                     %(gy)s,((dnnLayout_t*)PyArray_DATA(%(gy)s))[0],
                                      %(gm)s,((dnnLayout_t*)PyArray_DATA(%(gm)s))[0]);
        #endif
               status = dnnExecute_%(precision)s(pSplit, concat_res);
               if(0 != status)
               {
                   printf(\"ERROR:Split Execute\\n\"); 
                   exit(0);
               }
         #ifdef _DEBUG_
               printf(\" Split\\n\");  
         #endif    
               """ % sub
        return ccode 

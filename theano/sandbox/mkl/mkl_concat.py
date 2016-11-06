from theano.tensor import as_tensor_variable, TensorType
import theano
from theano.gof import Apply,HideC
from theano import gof
from theano.tensor.basic import Join, Split
from theano.tensor.nnet import mkldnn_helper
from theano.tensor.blas import ldflags

class JoinMKL4(gof.Op):
    """
    Join for MKL
    """
    def __init__(self,axis=1,uniq_id=1,lc=0):
       self.axis = axis
       self.uniq_id=uniq_id
       self.lc=lc

    def __eq__(self,other):
       return (type(self) == type(other) and
                (self.axis == other.axis) and
                (self.uniq_id == other.uniq_id))

    def make_node(self, x,y,xx,yy):
       return Apply(self,[x,y,xx,yy],[x.type()])

    def __hash__(self):
       return hash(type(self)) ^ hash(self.axis) ^ hash(self.uniq_id)

    def grad(self, inp, grads):
       x, y, xx, yy, = inp
       gz, = grads
       gx, gy, gxx, gyy = JoinMKLGrad4(self.axis,self.uniq_id)(x, y, xx, yy, gz)
       return gx, gy, gxx, gyy  

    def __str__(self):
       return "MklJoin4"

    def c_code_cache_version(self):
        return (1,0,hash(self.uniq_id))

    def c_headers(self):
        return super(JoinMKL4, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(JoinMKL4, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        final_code = mkldnn_helper.mkldnn_header_text()
        final_code += """
               //#define _DEBUG_
               static dnnPrimitive_t pConcat = NULL;
               static void* internal_ptr = NULL;
               static dnnLayout_t out_layer = NULL;
               static void *concat_res[dnnResourceNumber];
               static npy_intp out_dim[4] = {0};
                    """ 
        return final_code 

    def c_cleanup_code_struct(self,node, name):
        sub = {}
        sub['uid'] = self.uniq_id
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
             //std::cout<<"concat fwd cleaning up "<<%(uid)s<<std::endl;
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
             """ % sub 

    def c_code(self, node, name, inputs, out, sub):
        x, y, xx, yy, = inputs
        z, = out
        sub['axis'] = self.axis
        sub['z'] = z
        sub["L"] = 4 
        sub['uid'] = self.uniq_id
        sub['lc'] = self.lc
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
               //std::cout<<"concat fwd start "<<%(uid)s<<std::endl;
               int status = 0;
               if(NULL == pConcat)
               {
                  dnnLayout_t layout_int[%(L)s];
                  memcpy(out_dim, PyArray_DIMS(%(x)s), PyArray_NDIM(%(x)s)*sizeof(npy_intp));
              #ifdef _DEBUG_
                  for(int i=0; i < PyArray_NDIM(%(x)s);i++)
                        if(PyArray_DIMS(%(x)s)[i]<0)
                        {
                           printf(\"ERROR:%%d, %%d\\n\",i,PyArray_DIMS(%(x)s)[i]);
                           exit(0);
                        }
              #endif
                  out_dim[%(axis)s] = 0;
               """ % sub
        for i, inp in enumerate([x,y,xx,yy]):
            d={}
            d['i'] = i
            d['inp'] = inp
            d['axis'] = self.axis

            ccode += """
                  layout_int[%(i)s] = ((dnnLayout_t*)PyArray_DATA(%(inp)s))[0];
                  //std::cout<<"i "<<%(i)s<<" "<<layout_int[%(i)s]<<std::endl;
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
                   printf(\"Create InternalBuffer: %%x for concatelayer %%d, lc %%d \\n\",internal_ptr,%(uid)s,%(lc)s);
               #endif
               }
            """ % sub
        ccode += """
               if(NULL == %(z)s)
               {
               #ifdef _DEBUG_
                   printf(\"outDIM: %%d, %%d,%%d,%%d\\n\",out_dim[0],out_dim[1],out_dim[2],out_dim[3]);
               #endif
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
          #ifdef _DEBUG_
               printf("concatenate finish,%%d, %%d, %%d,%%d, %%x, uid: %%d\\n",PyArray_DIMS(%(z)s)[0],PyArray_DIMS(%(z)s)[1],
                  PyArray_DIMS(%(z)s)[2],PyArray_DIMS(%(z)s)[3], %(z)s, %(uid)s);
          #endif
               """ % sub
        return ccode 

class JoinMKLGrad4(gof.Op):
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

    def make_node(self, x,y, xx, yy, gz):
       return Apply(self,[x, y, xx, yy, gz],[x.type(),y.type(),xx.type(),yy.type()])

    def __hash__(self):
       return hash(type(self)) ^ hash(self.axis) ^ hash(self.uniq_id)

    def __str__(self):
       return "MklJoinGrad4_%d"%(self.uniq_id)

    def c_code_cache_version(self):
        return (1,0,hash(self.uniq_id))

    def c_headers(self):
        return super(JoinMKLGrad4, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(JoinMKLGrad4, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        final_code = mkldnn_helper.mkldnn_header_text()
        final_code += """
               //#define _DEBUG_
               static dnnPrimitive_t pSplit = NULL;
               static void *concat_res[dnnResourceNumber];
               void* internal_ptr0=NULL;
               void* internal_ptr1=NULL;
               void* internal_ptr2=NULL;
               void* internal_ptr3=NULL;
               """ 
        return final_code 

    def c_cleanup_code_struct(self,node, name):
        sub = {}
        sub['L'] = len(node.inputs)
        sub["uid"] = self.uniq_id
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
             //std::cout<<"concat bwd cleaning up "<<%(uid)s<<std::endl;
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
             if(internal_ptr0){
              dnnReleaseBuffer_F32(internal_ptr0);
             }else{
              std::cout<<"p0 null\\n";
             }
             if(internal_ptr1){
              dnnReleaseBuffer_F32(internal_ptr1);
             }else{
              std::cout<<"p1 null\\n";
             }
             if(internal_ptr2){
              dnnReleaseBuffer_F32(internal_ptr2);
             }else{
              std::cout<<"p2 null\\n";
             }
             if(internal_ptr3){
              dnnReleaseBuffer_F32(internal_ptr3);
             }else{
              std::cout<<"p3 null\\n";
             }
             """ % sub 

    def c_code(self, node, name, inputs, out, sub):
        x, y, xx, yy, gz, = inputs
        gx, gy, gxx, gyy, = out
        sub['axis'] = self.axis
        sub['gx'] = gx
        sub['gy'] = gy
        sub['gz'] = gz
        sub['gxx'] = gxx
        sub['gyy'] = gyy
        sub["L"] = 4 
        sub["uid"] = self.uniq_id
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
        sub['xx'] = xx
        sub['yy'] = yy
        ccode = """
               //printf(\"concatGrad start uid: %%d\\n\",%(uid)s);
               int status = 0;
               if(NULL == pSplit)
               {
                     size_t dstChannelSize[%(L)s] = {0};
                     dstChannelSize[0] = (size_t)PyArray_DIMS(%(x)s)[%(axis)s];
                     dstChannelSize[1] = (size_t)PyArray_DIMS(%(y)s)[%(axis)s];
                     dstChannelSize[2] = (size_t)PyArray_DIMS(%(xx)s)[%(axis)s];
                     dstChannelSize[3] = (size_t)PyArray_DIMS(%(yy)s)[%(axis)s];

                     status = dnnSplitCreate_%(precision)s(&pSplit, NULL, %(L)s,
                                        ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0],
                                         dstChannelSize);
                     if(0 != status)
                     {
                           printf(\"ERROR: dnnSplitCreate Primitive: %(L)s, layer:%%x, %%d, %%d, %%d, %%d\\n\",
                                        ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0],dstChannelSize[0],
                                        dstChannelSize[1],dstChannelSize[2],dstChannelSize[3]);
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
                      
              	      //create internal buffer
                      if(NULL == internal_ptr0)
                      {
                               status = dnnAllocateBuffer_%(precision)s(
                                               reinterpret_cast<void **>(&internal_ptr0),
                                               layout_int);
                           #ifdef _DEBUG_
                               printf(\"Create InternalBuffer: %%x\\n\",internal_ptr0);
                           #endif
                      }
                      int size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_int);
                      if(size != PyArray_DIMS(%(gx)s)[0]*PyArray_STRIDES(%(gx)s)[0])
                      {
                                printf(\"ERROR:int buffer Size: %%d != usr: %%d\\n\",size,
                                                PyArray_DIMS(%(gx)s)[0]*PyArray_STRIDES(%(gx)s)[0]);
                                exit(0);            
                      }
                      memset(internal_ptr0,0,size);
                      concat_res[dnnResourceMultipleDst + 0] = reinterpret_cast<void*>(internal_ptr0);
                      ((void**)PyArray_DATA(%(gx)s))[1] = internal_ptr0; 
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
                      //create internal buffer
                      if(NULL == internal_ptr1)
                      {
                               status = dnnAllocateBuffer_%(precision)s(
                                               reinterpret_cast<void **>(&internal_ptr1),
                                               layout_int);
                           #ifdef _DEBUG_
                               printf(\"Create InternalBuffer: %%x\\n\",internal_ptr1);
                           #endif
                      }
                      int size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_int);
                      if(size != PyArray_DIMS(%(gy)s)[0]*PyArray_STRIDES(%(gy)s)[0])
                      {
                                printf(\"ERROR:int buffer Size: %%d != usr: %%d\\n\",size,
                                                PyArray_DIMS(%(gy)s)[0]*PyArray_STRIDES(%(gy)s)[0]);
                                exit(0);
                      }
                      memset(internal_ptr1,0,size);
                      concat_res[dnnResourceMultipleDst + 1] = reinterpret_cast<void*>(internal_ptr1);
                      ((void**)PyArray_DATA(%(gy)s))[1] = internal_ptr1;
                }
               """ % sub
        ccode += """
                if(NULL == %(gxx)s)
                {
                     %(gxx)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(xx)s),
                                                           PyArray_DIMS(%(xx)s),
                                                           PyArray_TYPE(%(xx)s),
                                                           0);
                      if(NULL == %(gxx)s)
                      {
                            %(fail)s
                      }
                      dnnLayout_t layout_int = ((dnnLayout_t*)PyArray_DATA(%(xx)s))[0];
                      ((dnnLayout_t*)PyArray_DATA(%(gxx)s))[0] = layout_int;
                      //create internal buffer
                      if(NULL == internal_ptr2)
                      {
                               status = dnnAllocateBuffer_%(precision)s(
                                               reinterpret_cast<void **>(&internal_ptr2),
                                               layout_int);
                           #ifdef _DEBUG_
                               printf(\"Create InternalBuffer: %%x\\n\",internal_ptr2);
                           #endif
                      }
                      int size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_int);
                      if(size != PyArray_DIMS(%(gxx)s)[0]*PyArray_STRIDES(%(gxx)s)[0])
                      {
                                printf(\"ERROR:int buffer Size: %%d != usr: %%d\\n\",size,
                                                PyArray_DIMS(%(gxx)s)[0]*PyArray_STRIDES(%(gxx)s)[0]);
                                exit(0);
                      }
                      memset(internal_ptr2,0,size);
                      concat_res[dnnResourceMultipleDst + 2] = reinterpret_cast<void*>(internal_ptr2);
                      ((void**)PyArray_DATA(%(gxx)s))[1] = internal_ptr2;
                }
               """ % sub
        ccode += """
                if(NULL == %(gyy)s)
                {
                     %(gyy)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(yy)s),
                                                           PyArray_DIMS(%(yy)s),
                                                           PyArray_TYPE(%(yy)s),
                                                           0);
                      if(NULL == %(gyy)s)
                      {
                            %(fail)s
                      }
                      dnnLayout_t layout_int = ((dnnLayout_t*)PyArray_DATA(%(yy)s))[0];
                      ((dnnLayout_t*)PyArray_DATA(%(gyy)s))[0] = layout_int;
                      //create internal buffer
                      if(NULL == internal_ptr3)
                      {
                               status = dnnAllocateBuffer_%(precision)s(
                                               reinterpret_cast<void **>(&internal_ptr3),
                                               layout_int);
                           #ifdef _DEBUG_
                               printf(\"Create InternalBuffer: %%x\\n\",internal_ptr3);
                           #endif
                      }
                      int size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_int);
                      if(size != PyArray_DIMS(%(gxx)s)[0]*PyArray_STRIDES(%(gyy)s)[0])
                      {
                                printf(\"ERROR:int buffer Size: %%d != usr: %%d\\n\",size,
                                                PyArray_DIMS(%(gyy)s)[0]*PyArray_STRIDES(%(gyy)s)[0]);
                                exit(0);
                      }
                      memset(internal_ptr3,0,size);
                      concat_res[dnnResourceMultipleDst + 3] = reinterpret_cast<void*>(internal_ptr3);
                      ((void**)PyArray_DATA(%(gyy)s))[1] = internal_ptr3;
                }
               """ % sub
        ccode += """
               concat_res[dnnResourceSrc] = reinterpret_cast<void*>(((void**)PyArray_DATA(%(gz)s))[1]);
        #ifdef _DEBUG_ 
               printf(\"gx: %%x, gy: %%x, gxx: %%x, gyy: %%x\\n\",%(gx)s,%(gy)s,%(gxx)s,%(gyy)s);                             
               printf(\"split start: %%x, src:%%x, data: %%x, %%x, %%x, %%x\\n\",pSplit,concat_res[dnnResourceSrc],
                                        concat_res[dnnResourceMultipleDst + 3],
                                        concat_res[dnnResourceMultipleDst + 2],
                                        concat_res[dnnResourceMultipleDst + 1],
                                        concat_res[dnnResourceMultipleDst + 0]);
               printf(\"LayerOut:%%x, %%x, %%x,%%x, %%x, %%x, %%x, %%x\\n\",%(gx)s,((dnnLayout_t*)PyArray_DATA(%(gx)s))[0],
                                     %(gy)s,((dnnLayout_t*)PyArray_DATA(%(gy)s))[0],
                                      %(gxx)s,((dnnLayout_t*)PyArray_DATA(%(gxx)s))[0],
                                      %(gyy)s,((dnnLayout_t*)PyArray_DATA(%(gyy)s))[0]);

        #endif
               status = dnnExecute_%(precision)s(pSplit, concat_res);
               if(0 != status)
               {
                   printf(\"ERROR:Split Execute\\n\"); 
                   exit(0);
               }
         #ifdef _DEBUG_
               printf(\" Split Finished %%d\\n\",%(uid)s);  
         #endif    
               """ % sub
        return ccode 

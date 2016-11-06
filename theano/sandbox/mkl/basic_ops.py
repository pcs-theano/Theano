from theano.tensor import as_tensor_variable, TensorType
import theano
from theano import gof
from theano.gof import Apply, Op
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import mkl_helper

'''
class U2IBase(Op):
    def c_headers(self):
        return super(U2IBase, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(U2I, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        ccode = mkl_helper.header_text()
        ccode += """
            //#define _DEBUG_ 
            static void* internal_ptr = NULL; //mkl data buffer
            static dnnLayout_t layout_int = NULL;
            static dnnLayout_t layout_usr = NULL;
            static dnnPrimitive_t convert_to_int = NULL;
            static dnnPrimitive_t OpFwd = NULL; //Op Forward describe
            void *convert_resources[dnnResourceNumber];
        """
        return ccode 
'''

class U2I(gof.Op):
    def __init__(self, Op, target, params, uniq_id=None):
       self.params = params
       self.Op = Op
       self.target = target
       self.uniq_id = uniq_id

    def __hash__(self):
       return hash(type(self)) ^ hash(self.params) ^ hash(self.Op) ^ hash(self.target) ^ hash(self.uniq_id)

    def __eq__(self,other):
       return (type(self) == type(other) and 
                self.Op == other.Op and 
                self.params == other.params and
                self.target == other.target and 
                self.uniq_id == other.uniq_id)

    def __str__(self):
       return "U2I_%s"%(self.Op)

    def make_node(self,x):
       out = x.type()
       return gof.Apply(self,[x],[out]) 

    def c_code_cache_version(self):
        return (1,0, self.uniq_id)

    def c_headers(self):
        return super(U2I, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(U2I, self).c_compile_args()
        return compile_args

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [U2IGrad(self.Op,self.target,self.params, self.uniq_id)(x, gz)]
        #return [I2U()(gz)]

    def c_support_code(self):
        final_code = mkl_helper.header_text()
        final_code += """
                //    #define _DEBUG_ 
                    static void* internal_ptr = NULL; //mkl_dnn data buffer
                    static dnnLayout_t layout_int = NULL;
                    static dnnLayout_t layout_usr = NULL;
                    static dnnPrimitive_t convert_to_int = NULL;
                    static dnnPrimitive_t OpFwd = NULL; //Op Forward describe
                    void *convert_resources[dnnResourceNumber];
                    """
        return final_code 

    def c_cleanup_code_struct2(self,node, name):
        d={}
        if 'float32' == node.inputs[0].type.dtype:
            d['precision'] = "F32"
        elif "float64" == node.inputs[0].type.dtype:
            d['precision'] = "F64"
        else:
            raise Exception("Type %s not implemented" % 
                            node.inputs[0].type.dtype)
       
        return """ 
              int status = 0;
              if(NULL != internal_ptr)
              {
                     status = dnnReleaseBuffer_%(precision)s(internal_ptr);
                     if(0 != status)
                     {
                            printf(\"\\nERROR:dnnReleaseBuffer_%(precision)s free internal buffer\\n\");
                     }
                     internal_ptr = NULL;
              }

              if(NULL != OpFwd)
              {
                      status = dnnDelete_%(precision)s(OpFwd);
                      if(0 != status)
                      {
                            printf(\"\\nERROR:dnnDelete_%(precision)s OpFwd\\n\");
                      } 
                      OpFwd = NULL;
                      printf(\"Free OpFwd\\n\");
              }

              if(NULL != convert_to_int)
              {
                      status = dnnDelete_%(precision)s(convert_to_int);
                      if(0 != status)
                      {
                            printf(\"\\nERROR:dnnDelete_%(precision)s convert_to_int\\n\");
                      }
                      convert_to_int = NULL;
              }

              if(NULL != layout_usr)
              {
                      status = dnnLayoutDelete_%(precision)s(layout_usr);
                      if(0 != status)
                      {
                              printf(\"\\nERROR:dnnLayoutDelete__%(precision)s layout_usr in U2I\\n\");
                              exit(0);
                      }
                      layout_usr = NULL;
              }

              if(NULL != layout_int)
              {
                      status = dnnLayoutDelete_%(precision)s(layout_int);
                      if(0 != status)
                      {
                             printf(\"\\nERROR:dnnLayoutDelete__%(precision)s layout_int in U2I\\n\");
                             exit(0);
                      }
                      layout_int = NULL;
              }
              """ % d

    def c_init_code_struct(self, node, name, sub):
        if 'float32' == node.inputs[0].type.dtype:
            sub['precision'] = "F32"
        elif "float64" == node.inputs[0].type.dtype:
            sub['precision'] = "F64"
        else:
            raise  Exception("Type %s not implemented" %
                                node.inputs[0].type.dtype)
        sub['target'] = self.target
        N = self.params[0]
        C = self.params[1]
        H = self.params[2]
        W = self.params[3]

        sub['N'] = N
        sub['C'] = C
        sub['H'] = H
        sub['W'] = W
        sub['Op'] = self.Op
        print ("call ",self.Op)
        if self.Op  not in {'max':1,'average':1,'relu':1,'lrn':1,'conv':1}:
             print ("ERROR op Flag ",self.Op, ' is not in {maxPool,lrn,relu,conv,avgPool}')
             sys.exit(0)
        ccode = """
           size_t bottom_size[4] = {%(W)s,%(H)s,%(C)s,%(N)s};
           size_t bottom_stride[4] = {1,%(W)s,%(W)s*%(H)s,%(W)s*%(H)s*%(C)s};
           //create usr layerout
           int status = dnnLayoutCreate_%(precision)s(&layout_usr,
                                                     4,bottom_size,
                                                     bottom_stride);
           if(0 != status)
           {
                 printf(\"ERROR: create usr[%(N)s,%(C)s,%(H)s,%(W)s] in cvt_%(Op)s\\n\");
                 exit(0);
           }
        """ % sub

        if 'lrn' == self.Op:
            local_size = self.params[4]
            alpha = self.params[5]
            beta = self.params[6]
            k = self.params[7]
            sub['local_size'] = local_size
            sub['alpha'] = alpha
            sub['beta'] = beta
            sub['k'] = k
            
            ccode += """
            status = dnnLRNCreateForward_%(precision)s(&OpFwd,
                                         NULL,//attributes
                                         layout_usr,
                                         %(local_size)s,//size
                                         %(alpha)s,
                                         %(beta)s,
                                         %(k)s);
            if(0 != status)
            {
                 printf(\"ERROR:lrn Create lrn: %%x\\n\",OpFwd);
                 exit(0);
            }
            """ % sub
              
        elif 'relu' == self.Op:
            negative_slope = self.params[4]
            sub['negative_slope'] = negative_slope
            ccode += """
             status = dnnReLUCreateForward_%(precision)s(&OpFwd,
                                         NULL,///attributes,
                                         layout_usr,//group
                                         %(negative_slope)s);
             if(0 != status)
             {
                    printf(\"ERROR: relu Create relu: %%x\\n\",OpFwd);
                    exit(0);
             }
             """ % sub
            
        elif 'max' == self.Op:
            kH = self.params[4]
            kW = self.params[5]

            dH = self.params[6]
            dW = self.params[7]

            pH = self.params[8]
            pW = self.params[9]

            sub['dH'] = dH
            sub['dW'] = dW
            sub['pH'] = pH
            sub['pW'] = pW

            sub['kH'] = kH
            sub['kW'] = kW

            ccode += """
             int pad[2] = {- %(pW)s, - %(pH)s};
             size_t stride[2] = {%(dW)s, %(dH)s};
             size_t kernel_size[2] = {%(kW)s,%(kH)s};

             status = dnnPoolingCreateForward_%(precision)s(&OpFwd,//Bias
                                         NULL,///attributes,
                                         dnnAlgorithmPoolingMax, //algorithm
                                         layout_usr,//group
                                         kernel_size,
                                         stride,
                                         pad,
                                         dnnBorderZeros);
              if(0 != status)
              {
                    printf(\"ERROR: dnnGroupsConvolutionCreateForwardBias_%(precision)s: %%x\\n\",OpFwd);
                    exit(0);
              }
              """ % sub

        elif 'average' == self.Op:
            kH = self.params[4]
            kW = self.params[5]

            dH = self.params[6]
            dW = self.params[7]

            pH = self.params[8]
            pW = self.params[9]

            sub['dH'] = dH
            sub['dW'] = dW
            sub['pH'] = pH
            sub['pW'] = pW

            sub['kH'] = kH
            sub['kW'] = kW
            ccode += """
             int pad[2] = {- %(pW)s, - %(pH)s};
             size_t stride[2] = {%(dW)s, %(dH)s};
             size_t kernel_size[2] = {%(kW)s,%(kH)s};

             status = dnnPoolingCreateForward_%(precision)s(&OpFwd,//Bias
                                         NULL,///attributes,
                                         dnnAlgorithmPoolingAvg,//algorithm
                                         layout_usr,
                                         kernel_size,
                                         stride,
                                         pad,
                                         dnnBorderZeros);
              if(0 != status)
              {
                    printf(\"ERROR: dnnGroupsConvolutionCreateForwardBias_%(precision)s: %%x\\n\",OpFwd);
                    exit(0);
              }
              """ % sub

        elif 'conv' == self.Op:# and 4 == node.inputs[0].type.ndim:
            Oc = self.params[4] #weight
            Ic = self.params[5]
            kH = self.params[6]
            kW = self.params[7]

            if Ic != C:
                raise  Exception("Input channel %d not equal to filter channel:%d" % (Ic,C))

            dH = self.params[8]
            dW = self.params[9]
            pH = self.params[10]
            pW = self.params[11]
            group = self.params[12]

            oH = 1 + (H - kH + 2 * pH) / dH;
            oW = 1 + (W - kW + 2 * pW) / dW;

            sub['dH'] = dH
            sub['dW'] = dW
            sub['pH'] = pH
            sub['pW'] = pW

            sub['oC'] = Oc
            sub['Ic'] = Ic
            sub['kH'] = kH
            sub['kW'] = kW
            
            sub['oH'] = oH
            sub['oW'] = oW
            sub['group'] = group

            ccode += """
             int pad[2] = {- %(pW)s, - %(pH)s};
             size_t stride[2] = {%(dW)s, %(dH)s};

             //size_t weight_size[4] = {%(kW)s, %(kH)s, %(Ic)s, %(oC)s};
             size_t weight_size[5] = {%(kW)s, %(kH)s, %(Ic)s/%(group)s, %(oC)s/%(group)s, %(group)s};
             
             size_t top_size[4] = {%(oW)s, %(oH)s, %(oC)s, %(N)s};
             status = dnnGroupsConvolutionCreateForwardBias_%(precision)s(&OpFwd,//Bias
                                         NULL,///attributes,
                                         dnnAlgorithmConvolutionDirect,
                                         %(group)s,//group
                                         4,//dims
                                         bottom_size,
                                         top_size,
                                         weight_size,
                                         stride,
                                         pad,
                                         dnnBorderZeros);
              if(0 != status)
              {
                    printf(\"ERROR: dnnGroupsConvolutionCreateForwardBias_%(precision)s: %%x\\n\",OpFwd);
                    exit(0);
              }
             """ % sub

        ccode += """
              //create internal layerout
              status = dnnLayoutCreateFromPrimitive_%(precision)s(&layout_int,
                                                         OpFwd,
                                                         %(target)s);
              if(0 != status)
              {
                    printf(\"ERROR:dnnLayoutCreateFromPrimitive_%(precision)s\\n\");
                    exit(0);
              }

              size_t size = dnnLayoutGetMemorySize_%(precision)s(layout_usr);
              if(size != dnnLayoutGetMemorySize_%(precision)s(layout_int))
              {
                   printf(\"Warining:U2I Usr: %%d != int: %%d\\n\",dnnLayoutGetMemorySize_%(precision)s(layout_usr),
                                  dnnLayoutGetMemorySize_%(precision)s(layout_int));
                   printf(\"DIM: %(N)s,%(C)s,%(H)s,%(W)s\\n\");
              }
              if(NULL == convert_to_int)
              {
                   status = dnnConversionCreate_%(precision)s(&convert_to_int, layout_usr, layout_int);
                   if(0 != status)
                   {
                         printf(\"ERROR: dnnConversionCreate_%(precision)s \\n\");
                        exit(0);
                    }
              }
              """ % sub
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        sub['x'] = x
        sub['z'] = z
        sub['target'] = self.target
        if 'float32' == node.inputs[0].type.dtype:
            sub['type'] = "float"
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
        elif "float64" == node.inputs[0].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
        else:
            raise  Exception("Type %s not implemented" %
                                node.inputs[0].type.dtype)
        ccode = """
               if(NULL == OpFwd)
               {
                    printf(\"ERROR: Op OpFwd is NULL\\n\");
                    exit(0);
               }
               //U2I op
               int status = 0;
               if(NULL == %(z)s)
               {

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

                    //create PyArrayObject
                    %(z)s = (PyArrayObject*)PyArray_ZEROS(4, //U2I, owen
                                          PyArray_DIMS(%(x)s),
                                          PyArray_TYPE(%(x)s),
                                          0);
                    if(NULL == %(z)s)
                    {
                        %(fail)s
                    }
             } //
 
             if(layout_int != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0])
                   ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_int; //layout int
             if(internal_ptr != ((void**)PyArray_DATA(%(z)s))[1])
                   ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr;
            
             convert_resources[dnnResourceTo] = reinterpret_cast<void *>(internal_ptr);
             convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
             //cvt
             status = dnnExecute_%(precision)s(convert_to_int, convert_resources);
             if(0 != status)
             {
                 printf(\"ERROR:dnnExecute_%(precision)s\\n\");
                 exit(0);
             }
   #ifdef _DEBUG_
       printf(\"U2I:%%x to %%x\\n\",convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
       printf(\"Input: %%f \\n\",((float*)convert_resources[dnnResourceFrom])[2]); 
   #endif
       """ % sub
        return ccode 
        


class U2IGrad(gof.Op):
    def __init__(self, Op, target, params, uniq_id=None):
       self.params = params
       self.Op = Op
       self.target = target
       self.uniq_id = uniq_id

    def __hash__(self):
       return hash(type(self)) ^ hash(self.params) ^ hash(self.Op) ^ hash(self.target) ^ hash(self.uniq_id)

    def __eq__(self,other):
       return (type(self) == type(other) and
                self.Op == other.Op and
                self.params == other.params and
                self.target == other.target and
                self.uniq_id == other.uniq_id)

    def __str__(self):
       return "U2IGrad_%s"%(self.Op)

    def c_code_cache_version(self):
        return (1,0, self.uniq_id)

    def c_headers(self):
        return super(U2IGrad, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(U2IGrad, self).c_compile_args()
        return compile_args

    def make_node(self, x, gz):
        out = x.type()
        return gof.Apply(self,[x, gz],[out])

    def c_support_code(self):
        ccode = mkl_helper.header_text()
        ccode += """
         //#define _DEBUG_
         static void* internal_ptr = NULL;
         static void* usr_ptr = NULL;
         static dnnLayout_t layout_int = NULL;
         static dnnLayout_t layout_usr = NULL;
         static dnnPrimitive_t convert_from_int = NULL;
         static dnnPrimitive_t FilterFwd = NULL;
         void *convert_resources[dnnResourceNumber];
         """
        return ccode


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


class I2U(gof.Op):
    def __init__(self, uniq_id=None):
        self.uniq_id = uniq_id

    def __hash__(self):
       return hash(type(self)) ^ hash(self.uniq_id)

    def __eq__(self,other):
       return (type(self) == type(other) and
               self.uniq_id == other.uniq_id)

    def __str__(self):
       return "I2U"

    def c_code_cache_version(self):
        return (1,0, self.uniq_id)

    def c_headers(self):
        return super(I2U, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(I2U, self).c_compile_args()
        return compile_args

    def make_node(self, x):
        out = x.type()
        return gof.Apply(self,[x],[out])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [I2UGrad(self.uniq_id)(x, gz)]

    def c_support_code(self):
        ccode = mkl_helper.header_text()
        ccode += """ 
        // #define _DEBUG_
         static void* internal_ptr = NULL;
         static dnnLayout_t layout_int = NULL;
         static dnnLayout_t layout_usr = NULL;
         static dnnPrimitive_t convert_from_int = NULL;
         void *convert_resources[dnnResourceNumber];
         """
        return ccode

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
              #if 0
                float *out_p = (float*)((void**)PyArray_DATA(%(x)s))[1];
                printf(\"I2U fwd, x: %%g, %%g, %%g, %%g, %%g\\n\", out_p[0], out_p[1], out_p[2], out_p[3], out_p[4]);
                //printf(\"I2U fwd, x: %%x, %%x, %%x, %%x, %%x\\n\", out_p[0], out_p[1], out_p[2], out_p[3], out_p[4]);
              #endif

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

                  #if 0
                    printf(\"U2I, fwd, out_stride: %%d, %%d, %%d, %%d\\n\",out_stride[0],out_stride[1],out_stride[2],out_stride[3]);
                    printf(\"U2I, fwd, bottom size: %%d, %%d, %%d, %%d\\n\",bottom_size[0],bottom_size[1],bottom_size[2],bottom_size[3]);
                  #endif

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

                  //Get layerout and internal buffer from input.
                  layout_int = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];//get internal layerout
                  internal_ptr = ((void**)PyArray_DATA(%(x)s))[1];

                  status = dnnConversionCreate_%(precision)s(&convert_from_int, layout_int, layout_usr);
                  if(0 != status)
                  {
                        printf(\"ERROR:dnnConversionCreate_%(precision)s\\n\");
                        exit(0);
                  }

               }//out

               convert_resources[dnnResourceTo]=reinterpret_cast<void *>(PyArray_DATA(%(z)s));
               convert_resources[dnnResourceFrom] = reinterpret_cast<void *>(internal_ptr);

           #ifdef _DEBUG_
               printf(\"%%x,%%x,I2U %%x to %%x\\n\",convert_from_int,layout_int,convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
               printf(\"I2U %%f, %%f\\n\",((float*)PyArray_DATA(%(z)s))[1],((float*)PyArray_DATA(%(z)s))[3]);
               printf(\"DIMS: %%d, %%d, %%d, %%d\\n\",PyArray_DIMS(%(z)s)[0],PyArray_DIMS(%(z)s)[1],PyArray_DIMS(%(z)s)[2],PyArray_DIMS(%(z)s)[3]);
           #endif


               if(dnnLayoutGetMemorySize_%(precision)s(layout_int)!= PyArray_DIMS(%(z)s)[0]*PyArray_STRIDES(%(z)s)[0])
               {
                      printf(\"Error I2U: int %%d != usr %%d\\n\",dnnLayoutGetMemorySize_%(precision)s(layout_int),
                                                  PyArray_DIMS(%(z)s)[0]*PyArray_STRIDES(%(z)s)[0]);
 
                      if(PyArray_NDIM(%(z)s) ==4)
                               printf(\"DIMS %%d, %%d, %%d, %%d\\n\",PyArray_DIMS(%(z)s)[0],PyArray_DIMS(%(z)s)[1],
                                                  PyArray_DIMS(%(z)s)[2],PyArray_DIMS(%(z)s)[3]);
               //       exit(0);
               }

               //cvt
               status = dnnExecute_%(precision)s(convert_from_int, convert_resources);
               if(0 != status)
               {
                    printf(\"ERROR: convert_from_int in I2U\\n\");
                    exit(0);
               }
               """ % sub
        return ccode         



class I2UGrad(gof.Op):
    def __init__(self,uniq_id):
       self.uniq_id=uniq_id

    def __hash__(self):
       return hash(type(self)) ^ hash(self.uniq_id)

    def __eq__(self,other):
       return (type(self) == type(other) and self.uniq_id == other.uniq_id)

    def __str__(self):
       return "I2UGrad"

    def c_code_cache_version(self):
        return (1,0,self.uniq_id)

    def c_headers(self):
        return super(I2UGrad, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(I2UGrad, self).c_compile_args()
        return compile_args

    def make_node(self,x,gz):
        out = x.type()
        return gof.Apply(self,[x, gz],[out])

    def c_support_code(self):
        ccode = mkl_helper.header_text()
        ccode += """
         ///#define _DEBUG_
         static void* internal_ptr = NULL;
         static dnnLayout_t layout_int = NULL;
         static dnnLayout_t layout_usr = NULL;
         static dnnPrimitive_t convert_to_int = NULL;
         void *convert_resources[dnnResourceNumber];
         """
        return ccode

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

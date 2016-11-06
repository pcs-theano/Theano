from __future__ import absolute_import, print_function, division
import os
import logging

from six import integer_types

import theano
from theano import Apply
from theano import gof
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.blas_headers import blas_header_text
from theano.tensor.blas import ldflags
from theano.tensor.nnet import mkldnn_helper


class distribute(gof.Op):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        self.uniq_id = uniq_id

    def make_node(self, inp):
        inp = as_tensor_variable(inp)
        if inp.type.ndim != 4:
            raise TypeError('input must be 4D tensor')

        return Apply(self, [inp], [inp.type(), inp.type(), inp.type(), inp.type()])

    def c_support_code(self):
        ccode = mkldnn_helper.mkldnn_header_text()
        ccode += """
            #define _DEBUG_
            #define dimension 4
            #define CHECK_ERR(f, err) \\
                do { \\
                    (err) = (f); \\
                    if ((err) != E_SUCCESS) { \\
                        printf("Error in file [%s:%d], err code (%d)", __FILE__, __LINE__, err); \\
                    } \\
                } while(0) 
         static dnnLayout_t  int_layout_input_from_previous = NULL;
         void *z1_buf = NULL;
         void *z2_buf = NULL;
         void *z3_buf = NULL;
         void *z4_buf = NULL;
         void *input_buf = NULL;
        """
        return ccode

    def c_code(self, node, name, inp, out, sub):
        inp, = inp
        z1, z2, z3, z4, = out
        fail = sub['fail']

        if node.inputs[0].dtype == "float32":
            precision = 'F32'
            dtype = 'float'
        elif node.inputs[0].dtype == "float64":
            precision = 'F64'
            dtype = 'double'

        ccode = """
            
            int_layout_input_from_previous = ((dnnLayout_t*)PyArray_DATA(%(inp)s))[0];
            input_buf = ((void**)PyArray_DATA(%(inp)s))[1];
            //std::cout<<"dis fwd\\n";
            
            int err =0 ;
            if ( !(%(z1)s))
            {
                %(z1)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(inp)s),
                                                        PyArray_DIMS(%(inp)s),
                                                        PyArray_TYPE(%(inp)s),
                                                        0);
                if (NULL == %(z1)s)
                {
                    std::cout<<"z a f\\n";
                }
            }
            if ( !(%(z2)s))
            {
                %(z2)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(inp)s),
                                                        PyArray_DIMS(%(inp)s),
                                                        PyArray_TYPE(%(inp)s),
                                                        0);
                if (NULL == %(z2)s)
                {
                    std::cout<<"z a f\\n";
                }
            }
            if ( !(%(z3)s))
            {
                %(z3)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(inp)s),
                                                        PyArray_DIMS(%(inp)s),
                                                        PyArray_TYPE(%(inp)s),
                                                        0);
                if (NULL == %(z3)s)
                {
                    std::cout<<"z a f\\n";
                }
            }
            if ( !(%(z4)s))
            {
                %(z4)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(inp)s),
                                                        PyArray_DIMS(%(inp)s),
                                                        PyArray_TYPE(%(inp)s),
                                                        0);
                if (NULL == %(z4)s)
                {
                    std::cout<<"z a f\\n";
                }
            }

            //std::cout<<"l fo p "<<std::hex<<int_layout_input_from_previous<<std::endl;

            ((dnnLayout_t*)PyArray_DATA(%(z1)s))[0] = int_layout_input_from_previous;
            ((void**)PyArray_DATA(%(z1)s))[1] = input_buf;

            ((dnnLayout_t*)PyArray_DATA(%(z2)s))[0] = int_layout_input_from_previous;
            ((void**)PyArray_DATA(%(z2)s))[1] = input_buf;

            ((dnnLayout_t*)PyArray_DATA(%(z3)s))[0] = int_layout_input_from_previous;
            ((void**)PyArray_DATA(%(z3)s))[1] = input_buf;

            ((dnnLayout_t*)PyArray_DATA(%(z4)s))[0] = int_layout_input_from_previous;
            ((void**)PyArray_DATA(%(z4)s))[1] = input_buf;
    
            //printf(\"distribute: %%x, %%x, %%x, %%x\\n\",%(z1)s,%(z2)s,%(z3)s,%(z4)s);

        
        """ % locals() 

        return ccode

    def grad(self, inp, grads):
        inp, = inp
        gz1,gz2,gz3,gz4, = grads
        #print ("grad \n",inp,gz1,gz2,gz3,gz4)

        d_inp = distributeGrad(self.uniq_id)(gz1, gz2, gz3, gz4, inp)
        return [d_inp]

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)


class distributeGrad(gof.Op):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        self.uniq_id = uniq_id

    def make_node(self, inp1, inp2, inp3, inp4,x):
        inp1 = as_tensor_variable(inp1)
        inp2 = as_tensor_variable(inp2)
        inp3 = as_tensor_variable(inp3)
        inp4 = as_tensor_variable(inp4)

        if inp1.type.ndim != 4:
            raise TypeError('input must be 4D tensor')

        return Apply(self, [inp1,inp2,inp3,inp4,x], [inp1.type()])

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)

    def c_support_code(self):
        ccode = mkldnn_helper.mkldnn_header_text()
        ccode += """
            #define _DEBUG_
            static dnnLayout_t  int_layout_input_from_previous;
            #define CHECK_ERR(f, err) \\
                do { \\
                    (err) = (f); \\
                    if ((err) != E_SUCCESS) { \\
                        printf("Error in file [%s:%d], err code (%d)", __FILE__, __LINE__, err); \\
                    } \\
                } while(0) 
            static void *gz1_buf = NULL;
            static void *gz2_buf = NULL;
            static void *gz3_buf = NULL;
            static void *gz4_buf = NULL;
            static void* aggregated_output_buffer=NULL;
        """
        return ccode

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        return """
        //std::cout<<"releasing buffer\\n";
        dnnReleaseBuffer_F32(aggregated_output_buffer);
        """

    def c_code(self, node, name, inp, out, sub):
        gz1, gz2, gz3, gz4, x, = inp
        outgrad, = out
        fail = sub['fail']

        if node.inputs[0].dtype == "float32":
            dtype = 'float'
            precision = 'F32'
        elif node.inputs[0].dtype == "float64":
            dtype = 'double'
            precision = 'F64'
        ccode = """
            int err = 0;
            
            //printf(\"dis bwd gz1: %%x, gz2: %%x, gz3: %%x, gz4: %%x\\n\", %(gz1)s, %(gz2)s, %(gz3)s, %(gz4)s);

            if ( !(%(outgrad)s))
            {
                %(outgrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),
                                                        PyArray_DIMS(%(x)s),
                                                        PyArray_TYPE(%(x)s),
                                                        0);
                if (NULL == %(outgrad)s)
                {
                    std::cout<<"fdb\\n";
                }
            }


            dnnLayout_t gz1_layout = ((dnnLayout_t*)PyArray_DATA(%(gz1)s))[0];
            dnnLayout_t gz2_layout = ((dnnLayout_t*)PyArray_DATA(%(gz2)s))[0];
            dnnLayout_t gz3_layout = ((dnnLayout_t*)PyArray_DATA(%(gz3)s))[0];
            dnnLayout_t gz4_layout = ((dnnLayout_t*)PyArray_DATA(%(gz4)s))[0];
            #if 0
            std::cout<<"################################################\\n";
            std::cout<<"################################################\\n";
            std::cout<<"################################################\\n";
            std::cout<<std::hex<<"l1 "<<gz1_layout<<" l2 "<<gz2_layout<<" l3 "<<gz3_layout<<" l4 "<<gz4_layout<<std::endl;
            std::cout<<"################################################\\n";
            std::cout<<"################################################\\n";
            std::cout<<"################################################\\n";
            #endif
            gz1_buf = ((void**)PyArray_DATA(%(gz1)s))[1];
            gz2_buf = ((void**)PyArray_DATA(%(gz2)s))[1];
            gz3_buf = ((void**)PyArray_DATA(%(gz3)s))[1];
            gz4_buf = ((void**)PyArray_DATA(%(gz4)s))[1];

            int buffer_size = dnnLayoutGetMemorySize_F32(gz1_layout);

            if (NULL == aggregated_output_buffer) {
                int e = dnnAllocateBuffer_F32(&aggregated_output_buffer, gz1_layout);
                if (E_SUCCESS != e){
                  std::cout<<"dis allocate fail with error code "<<e<<std::endl;
                }       
                memset(aggregated_output_buffer,0,buffer_size);  
            }

            ((dnnLayout_t*)PyArray_DATA(%(outgrad)s))[0] = gz1_layout;
            ((void**)PyArray_DATA(%(outgrad)s))[1] = aggregated_output_buffer;
            #pragma omp parallel for
            #pragma ivdep
            for(int i=0;i<buffer_size/sizeof(float);i++){
                //std::cout<<"aggregating "<<i<<std::endl;
                ((float*)aggregated_output_buffer)[i] = ((float*)gz1_buf)[i]+((float*)gz2_buf)[i]+((float*)gz3_buf)[i]+((float*)gz4_buf)[i];
            }

            //printf(\"distributeGrad: %%x\\n\",%(outgrad)s);
    
        """ % locals()
        return ccode


###############################################################################################################
###################################### dist 2 #########################################################################
###############################################################################################################
class distribute2(gof.Op):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        self.uniq_id = uniq_id

    def make_node(self, inp):
        inp = as_tensor_variable(inp)
        if inp.type.ndim != 4:
            raise TypeError('input must be 4D tensor')
        print ("make Node iv2")
        return Apply(self, [inp], [inp.type(), inp.type()])

    def c_support_code(self):
        ccode = mkldnn_helper.mkldnn_header_text()
        ccode += """
            #define _DEBUG_
            #define dimension 4
            #define CHECK_ERR(f, err) \\
                do { \\
                    (err) = (f); \\
                    if ((err) != E_SUCCESS) { \\
                        printf("Error in file [%s:%d], err code (%d)", __FILE__, __LINE__, err); \\
                    } \\
                } while(0) 
         static dnnLayout_t  int_layout_input_from_previous = NULL;
         void *z1_buf = NULL;
         void *z2_buf = NULL;
         void *input_buf = NULL;
        """
        return ccode

    def c_code(self, node, name, inp, out, sub):
        inp, = inp
        z1, z2,  = out
        fail = sub['fail']

        if node.inputs[0].dtype == "float32":
            precision = 'F32'
            dtype = 'float'
        elif node.inputs[0].dtype == "float64":
            precision = 'F64'
            dtype = 'double'

        ccode = """
            
            int_layout_input_from_previous = ((dnnLayout_t*)PyArray_DATA(%(inp)s))[0];
            input_buf = ((void**)PyArray_DATA(%(inp)s))[1];
            //std::cout<<"dis2 fwd\\n";
            
            int err =0 ;
            if ( !(%(z1)s))
            {
                %(z1)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(inp)s),
                                                        PyArray_DIMS(%(inp)s),
                                                        PyArray_TYPE(%(inp)s),
                                                        0);
                if (NULL == %(z1)s)
                {
                    std::cout<<"z a f\\n";
                }
            }
            if ( !(%(z2)s))
            {
                %(z2)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(inp)s),
                                                        PyArray_DIMS(%(inp)s),
                                                        PyArray_TYPE(%(inp)s),
                                                        0);
                if (NULL == %(z2)s)
                {
                    std::cout<<"z a f\\n";
                }
            }

            //std::cout<<"l fo p "<<std::hex<<int_layout_input_from_previous<<std::endl;

            ((dnnLayout_t*)PyArray_DATA(%(z1)s))[0] = int_layout_input_from_previous;
            ((void**)PyArray_DATA(%(z1)s))[1] = input_buf;

            ((dnnLayout_t*)PyArray_DATA(%(z2)s))[0] = int_layout_input_from_previous;
            ((void**)PyArray_DATA(%(z2)s))[1] = input_buf;

    
            //printf(\"distribute2 finished: %%x, %%x\\n\",%(z1)s,%(z2)s);

        
        """ % locals() 

        return ccode

    def grad(self, inp, grads):
        x, = inp
        gz1,gz2, = grads
        d_inp = distributeGrad2(self.uniq_id)(gz1, gz2, x)
        return [d_inp]

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)


class distributeGrad2(gof.Op):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        self.uniq_id = uniq_id

    def make_node(self, inp1, inp2, x):
        inp1 = as_tensor_variable(inp1)
        inp2 = as_tensor_variable(inp2)
        x = as_tensor_variable(x)
        if inp1.type.ndim != 4:
            raise TypeError('input must be 4D tensor')
        return Apply(self, [inp1,inp2,x], [inp1.type()])

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)

    def c_support_code(self):
        ccode = mkldnn_helper.mkldnn_header_text()
        ccode += """
            #define _DEBUG_
            static dnnLayout_t  int_layout_input_from_previous;
            #define CHECK_ERR(f, err) \\
                do { \\
                    (err) = (f); \\
                    if ((err) != E_SUCCESS) { \\
                        printf("Error in file [%s:%d], err code (%d)", __FILE__, __LINE__, err); \\
                    } \\
                } while(0) 
            static void *gz1_buf = NULL;
            static void *gz2_buf = NULL;
            static void* aggregated_output_buffer=NULL;
            static dnnLayout_t gz1_layout = NULL;
            static dnnLayout_t gz2_layout = NULL;
        """
        return ccode

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        return """
        //std::cout<<"releasing buffer\\n";
        dnnReleaseBuffer_F32(aggregated_output_buffer);
        """

    def c_code(self, node, name, inp, out, sub):
        gz1, gz2, x, = inp
        outgrad, = out
        fail = sub['fail']

        if node.inputs[0].dtype == "float32":
            dtype = 'float'
            precision = 'F32'
        elif node.inputs[0].dtype == "float64":
            dtype = 'double'
            precision = 'F64'
        ccode = """
            int err = 0;
            
            //printf(\"dis 2 bwd gz1: %%x, gz2: %%x\\n\", %(gz1)s, %(gz2)s);

            if ( !(%(outgrad)s))
            {
                %(outgrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),
                                                        PyArray_DIMS(%(x)s),
                                                        PyArray_TYPE(%(x)s),
                                                        0);
                if (NULL == %(outgrad)s)
                {
                    std::cout<<"fdb\\n";
                }
            }

            gz1_layout = ((dnnLayout_t*)PyArray_DATA(%(gz1)s))[0];
            gz2_layout = ((dnnLayout_t*)PyArray_DATA(%(gz2)s))[0];

            #if 0
            std::cout<<"################################################\\n";
            std::cout<<"################################################\\n";
            std::cout<<"################################################\\n";
            std::cout<<std::hex<<"l1 "<<gz1_layout<<" l2 "<<gz2_layout<<std::endl;
            std::cout<<"################################################\\n";
            std::cout<<"################################################\\n";
            std::cout<<"################################################\\n";
            #endif
            gz1_buf = ((void**)PyArray_DATA(%(gz1)s))[1];
            gz2_buf = ((void**)PyArray_DATA(%(gz2)s))[1];


            int buffer_size = dnnLayoutGetMemorySize_F32(gz1_layout);

            if (NULL == aggregated_output_buffer) {
                int e = dnnAllocateBuffer_F32(&aggregated_output_buffer, gz1_layout);
                if (E_SUCCESS != e){
                  std::cout<<"dis allocate fail with error code "<<e<<std::endl;
                }       
                memset(aggregated_output_buffer,0,buffer_size);  
            }

            ((dnnLayout_t*)PyArray_DATA(%(outgrad)s))[0] = gz1_layout;
            ((void**)PyArray_DATA(%(outgrad)s))[1] = aggregated_output_buffer;
            #pragma omp parallel for
            #pragma ivdep
            for(int i=0;i<buffer_size/sizeof(float);i++){
                //std::cout<<"aggregating "<<i<<std::endl;
                ((float*)aggregated_output_buffer)[i] = ((float*)gz1_buf)[i]+((float*)gz2_buf)[i];
            }

            //printf(\"distributeGrad2: %%x\\n\",%(outgrad)s);
    
        """ % locals()
        return ccode

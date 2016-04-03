import logging
import numpy
import theano
from theano import OpenMPOp, config
from theano.gof import Apply
from theano import gof
import theano

"""
   update paramter  Ops
"""
class ParamUpdate(gof.Op):
    def __init__(self,index,dim,shape,lr):
        self.index=index
        self.dim=dim
	self.shape=shape
	self.lr=lr

    def __hash__(self):
	return  (hash(type(self)) ^ hash(self.index) ^ hash(self.dim) ^
          hash(self.shape) ^ hash(self.lr))

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.index == other.index and
		self.dim== other.dim and
		self.shape == other.shape and
		self.lr == other.lr)

    def make_node(self, x,y):
        # Validate the inputs' type
        if x.type.ndim != y.type.ndim:
            raise TypeError('x ,y must have the same dim')
        #Create an output variable of the same type as x
        output_var = x.type()
        return gof.Apply(self, [x,y], [output_var])

    def c_code_cache_version(self):
        return (1, 0)

    def __str__(self):
	return "updataOp%d"%(self.index)

    def c_headers(self):
	return ['PcsParameter.hpp','<stdlib.h>','<omp.h>','<pthread.h>']

    def c_libraries(self):
	return ['PcsParameter','hiredis','pthread']

    def c_init_code(self):
	return ['if(0==c)\n\tc = (redisContext*)redisConnect((char*)\"%s\", %d);\n'%
                (config.ipxx,config.port)]

    def c_support_code(self):
	d={}
	d['index']=self.index
	dim=len(self.shape)
	d['ip']=config.ipxx
	d['port']=config.port
	ccode="""
	    static int ThreadCount=0;
	    static int WriteCount=0;
	    static int RWFlag=0;
	"""
	if 1==dim:
	    d['one']=self.shape[0]
	    ccode +="""
                void* WriteThread(void*output)
                {
		    redisContext* tmp = 
                        (redisContext*)redisConnect((char*)\"%(ip)s\",%(port)s);
                    while(tmp->err)
                    {
                        redisFree(tmp);
                        printf(\"redis Connect error\\n\");
                        sleep(1);
                        tmp = (redisContext*)redisConnect(
                                    (char*)\"%(ip)s\",%(port)s);
                    }

                    WriteArray(tmp,\"%(index)s\",%(one)s,(float*)output);
		    redisFree(tmp);
		    ThreadCount--;
		    pthread_exit(NULL);
                }
                void* ReadThread(void*output)
                {
                    redisContext* tmp = 
                        (redisContext*)redisConnect((char*)\"%(ip)s\",%(port)s);
                    while(tmp->err)
                    {
                        redisFree(tmp);
                        printf(\"redis Connect error\\n\");
                        sleep(1);
                        tmp = (redisContext*)redisConnect(
                                    (char*)\"%(ip)s\",%(port)s);
                    }

		    ReadArray(tmp,\"%(index)s\",%(one)s,(float*)output);
                    redisFree(tmp);
		    ThreadCount--;
		    pthread_exit(NULL);
                }
		""" % d
	elif 2==dim:
            d['one']=self.shape[0]
	    d['two']=self.shape[0]
            ccode +="""
                void* WriteThread(void*output)
                {
		    redisContext* tmp = 
                        (redisContext*)redisConnect((char*)\"%(ip)s\",%(port)s);
                    while(tmp->err)
		    {
                        redisFree(tmp);
                	printf(\"redis Connect error\\n\");
                	sleep(1);
			tmp = (redisContext*)redisConnect(
                                (char*)\"%(ip)s\",%(port)s);
        	    }
		    if(0 ==output)
		    {
		        printf(\"error:%(one)s,%(two)s \\n\");
		    }
		    WriteArray(tmp,\"%(index)s\",%(one)s,%(two)s,(float*)output);
                    redisFree(tmp);
		    ThreadCount--;
		    pthread_exit(NULL);
		    return 0;
                }
                void* ReadThread(void*output)
                {
		    redisContext* tmp = 
                        (redisContext*)redisConnect((char*)\"%(ip)s\",%(port)s);
                    while(tmp->err)
                    {
                        redisFree(tmp);
                        printf(\"redis Connect error\\n\");
                        sleep(1);
                        tmp = (redisContext*)redisConnect(
                                (char*)\"%(ip)s\",%(port)s);
                    }
                    if(0 ==output)
                    {
                        printf(\"error:%(one)s,%(two)s \\n\");
                    }
                    ReadArray(tmp,\"%(index)s\",%(one)s,%(two)s,(float*)output);
                    redisFree(tmp);
		    ThreadCount--;
		    pthread_exit(NULL);
		    return 0;
                }
                """ % d
	return ccode

    def c_code(self,node,name,inp,out,sub):
	fail=sub['fail']
	x,y,=inp
	z,=out
	d={}
	d['x']=x
	d['y']=y
	d['index']=self.index
	d['z']=z
	d['dim']=self.dim
	d['fail']=fail
	d['ip']=config.ipxx
	d['port']=config.port
	d['lr']=self.lr
	ccode="""
            static pthread_t threads[2];
		
	    if(PyArray_NDIM(%(x)s) != %(dim)s)
	    {
	        PyErr_SetString(PyExc_ValueError, "x must be %(dim)s D");
	        %(fail)s;
	    }
            if(PyArray_NDIM(%(y)s) != %(dim)s)
            {
                PyErr_SetString(PyExc_ValueError, " y must has %(dim)s D");
                %(fail)s;
            }
            if(0 == %(z)s)
            {
	        PyArray_Dims outshape;
	        outshape.ptr=PyArray_DIMS(%(x)s);
	        outshape.len=%(dim)s;
		%(z)s = (PyArrayObject*)PyArray_Newshape(
                            (PyArrayObject*)%(x)s,&outshape,NPY_CORDER);
	    }
            float*grad=(float*)PyArray_DATA(%(y)s);
	    float*parameter=(float*)PyArray_DATA(%(x)s);
            float*output =(float*)PyArray_DATA(%(z)s);
	    int ret=0;
	""" % d
	if self.dim ==1:
	    d['one']=self.shape[0]
	    ccode += """
	        {
		    #pragma omp parallel for
		    for(int i=0;i<%(one)s;i++)
		    {
		        output[i]=parameter[i]-%(lr)s*grad[i];
		    }
		    if((WriteCount==100 && 0 ==ThreadCount) || 0 == RWFlag)
		    {
		        ret == ReadArray(c,\"%(index)s\",%(one)s,output);
		        if(ret==0)
		        {
			    WriteCount=0;
			    RWFlag=1;
			}
			else
			    WriteCount--;
		    }

                    if(0 == ThreadCount)
		    {
                        ret = pthread_create(&threads[1], NULL,
                                    WriteThread,(void*)output);
                        if(ret)
                        {
                            printf(\"Create ReadThread for %(index)s error\\n\");
                        }
			ThreadCount++;
			WriteCount++;
		    }
	        }
		""" % d
	elif 2 == self.dim:
            d['one']=self.shape[0]
	    d['two']=self.shape[1]
	    ccode+="""
	        {
		    #pragma omp parallel for
		    for(int i=0;i<%(one)s*%(two)s;i++)
		    {
		        output[i]=parameter[i]-%(lr)s*grad[i];
		    }
		    if((WriteCount==3000 && 0 ==ThreadCount) || 0 == RWFlag)
		    {
			ret=ReadArray(c,\"%(index)s\",%(one)s,%(two)s,(float*)output);
			if(ret ==0)
			{
			    WriteCount=0;
			    RWFlag=1;
		        }
			else
			    WriteCount--;
		    }
		    if(ThreadCount ==0)
		    {
                        ret = pthread_create(&threads[1], NULL,WriteThread,(void*)output);
                        if(ret)
                        {
                            printf(\"Create WriteThread for %(index)s error\\n\");
                        }
		        ThreadCount++;
			WriteCount++;
		    }
		}
	    """ % d
	return ccode


class GetParam(gof.Op):
	def __init__(self,index,shape):
            self.index=index
	    self.shape=shape
	def __hash__(self):
	    return  hash(type(self)) ^ hash(self.index) ^ hash(self.shape)
	def __eq__(self,other):
	    return (type(self) == type(other) and
		self.index == other.index and
		self.shape == other.shape)
	def make_node(self, x):
            output_var=x.type()
	    return gof.Apply(self, [x], [output_var])
	def c_code_cache_version(self):
	    return (1,0)
	def __str__(self):
	    return "GetParamOp%d"%(self.index)
	def c_headers(self):
	    return ['PcsParameter.hpp','<stdlib.h>','<omp.h>']
	def c_libraries(self):
	    return ['PcsParameter','hiredis']
	def c_init_code(self):
	    return ['if(0==c)\n\tc = (redisContext*)redisConnect((char*)\"%s\", %d);\n'%
                    (config.ipxx,config.port)]

	def c_code(self,node,name,inp,out,sub):
		fail=sub['fail']
		x,=inp
		z,=out
		d={}
		d['x']=x
		d['z']=z
		d['fail']=fail
		d['index']=self.index
		dim=len(self.shape)
		d['dim']=dim
		ccode="""
                    if(PyArray_NDIM(%(x)s) != %(dim)s)
                    {
                       PyErr_SetString(PyExc_ValueError, "x must be %(dim)s D");
                       %(fail)s;
                    }
                    if(0 == %(z)s)
                    {
                       PyArray_Dims outshape;
                       outshape.ptr=PyArray_DIMS(%(x)s);
                       outshape.len=%(dim)s;
                       %(z)s = (PyArrayObject*)PyArray_Newshape(
                                    (PyArrayObject*)%(x)s,&outshape,NPY_CORDER);
                    }
		    float*output= (float*)PyArray_DATA(%(z)s);
		    {
		""" % d
		if dim ==1:
	            d['one']=self.shape[0]
		    ccode +="""
		        ReadArray(c,\"%(index)s\",%(one)s,output);
		    """ % d
		elif dim ==2:
		    d['one']=self.shape[0]
		    d['two']=self.shape[1]
		    ccode +="""
		        ReadArray(c,\"%(index)s\",%(one)s,%(two)s,output);
		    """ % d
		ccode+="""} """
		return ccode


class SetParam(gof.Op):
        def __init__(self,index,shape):
            self.index=index
            self.shape=shape
        def __hash__(self):
            return hash(type(self)) ^ hash(self.index) ^ hash(self.shape)
        def __eq__(self,other):
            return (type(self) == type(other) and
                self.index == other.index and
                self.shape == other.shape)
        def make_node(self, x):
            output_var=x.type()
            return gof.Apply(self, [x], [output_var])
        def c_code_cache_version(self):
            return (1,0)
        def __str__(self):
            return "SetParamOp%d"%(self.index)
        def c_headers(self):
            return ['PcsParameter.hpp','<stdlib.h>','<omp.h>']
        def c_libraries(self):
            return ['PcsParameter','hiredis']
	def c_support_code(self):
            d={}
            d['index']=self.index
            dim=len(self.shape)
            d['ip']=config.ipxx
            d['port']=config.port
            ccode="""
                static int ThreadCount=0;
            """
            if 1==dim:
	        d['one']=self.shape[0]
                ccode +="""
                    void* WriteThread(void*output)
                    {
                        redisContext* tmp = (redisContext*)redisConnect(
                                                (char*)\"%(ip)s\",%(port)s);
                        while(tmp->err)
                        {
                            redisFree(tmp);
                            printf(\"redis Connect error\\n\");
                            sleep(1);
                            tmp = (redisContext*)redisConnect(
                                        (char*)\"%(ip)s\",%(port)s);
                        }

                        WriteArray(tmp,\"%(index)s\",%(one)s,(float*)output);
                        redisFree(tmp);
                        ThreadCount--;
                        pthread_exit(NULL);
                    }
		""" % d
	    elif 2==dim:
	        d['one']=self.shape[0]
		d['two']=self.shape[1]
                ccode +="""
                    void* WriteThread(void*output)
                    {
                        redisContext* tmp = (redisContext*)redisConnect(
                                                (char*)\"%(ip)s\",%(port)s);
                        while(tmp->err)
                        {
                            redisFree(tmp);
                            printf(\"redis Connect error\\n\");
                            sleep(1);
                            tmp = (redisContext*)redisConnect(
                                        (char*)\"%(ip)s\",%(port)s);
                        }
                        if(0 ==output)
                        {
                            printf(\"error:%(one)s,%(two)s \\n\");
                        }
                        WriteArray(tmp,\"%(index)s\",%(one)s,%(two)s,
                                   (float*)output);
                        redisFree(tmp);
                        ThreadCount--;
                        pthread_exit(NULL);
                        return 0;
                    }
                """ % d
		return ccode

	def c_init_code(self):
            return ['if(0==c)\n\tc = (redisContext*)redisConnect((char*)\"%s\", %d);\n'%
                    (config.ipxx,config.port)]

        def c_code(self,node,name,inp,out,sub):
            fail=sub['fail']
            x,=inp
	    z,=out
            d={}
            d['x']=x
            d['index']=self.index
            dim=len(self.shape)
            d['dim']=dim
	    d['fail']=fail
            if not config.ipxx or not config.port:
                print "There is not configure paramter server Ip proxy and port\n"
                sys.exit(0)
	    d['ip']=config.ipxx
	    d['port']=config.port
	    if 1==dim:
		d['len']=self.shape[0]
            elif 2==dim:
		d['len']=self.shape[0]*self.shape[1]
            ccode="""
	        static pthread_t threads[2];
		int ret=0;
                if(PyArray_NDIM(%(x)s) != %(dim)s)
                {
                   PyErr_SetString(PyExc_ValueError, "x must be %(dim)s D");
                   %(fail)s;
                }
		float*param= (float*)PyArray_DATA(%(x)s);
                if(ThreadCount ==0)
                {
		    float output[%(len)s]={0.0};	
		    memcpy(output,param,%(len)s*sizeof(float));
		    printf(\"%%f \\n\",output[0]);
		""" % d
	    if 1 ==dim:
	        d['one']=self.shape[0]
	        ccode+="""
		   WriteArray(c,\"%(index)s\",%(one)s,(float*)output);
		   } 
	        """ % d
	    elif 2== dim:
	        d['one']=self.shape[0]
	        d['two']=self.shape[1]
	        ccode+="""
	            WriteArray(c,\"%(index)s\",%(one)s,%(two)s,(float*)output);
		    }
		""" % d
	    ccode +="""
	        float value=0.0;
	        ReadItem(c,\"%(index)s\",0,0,&value);
	        printf(\"v:%%f\\n\",value);
            """ % d
	    return ccode


class UpParam(gof.Op):
    def __init__(self,num,key=''):
	self.key=key
	self.num=num
    def __hash__(self):
        return hash(type(self)) ^ hash(self.key) ^ hash(self.num)
    def __eq__(self,other):
        return (type(self) == type(other) and
	        self.key == other.key and
		self.num == other.num)
    def make_node(self, x):
	output_var=x[0].type()
        return gof.Apply(self,x,[output_var])
    def c_code_cache_version(self):
        return (1,0)
    def __str__(self):
        return "UpParam(%s,%d)"%(self.key,self.num)
    def c_headers(self):
        return ['PcsParameter.hpp','time.h']
    def c_libraries(self):
        return ['PcsParameter','hiredis','pthread']
    def c_compile_args(self):
        return ['-O0','-g']
    def c_no_compile_args(self):
        return ['-O3','-openmp','-O2','-O3']
    def c_code(self,node,name,inp,out,sub):
        fail=sub['fail']
	d={}
        z,=out
	d['z']=z
        if not config.ipxx or not config.port:
            print "There is not configure paramter server Ip proxy and port\n"
            sys.exit(0)
	d['ip']=config.ipxx
	d['port']=config.port
	d['key']=self.key
        d['fail']=fail
	d['num']=self.num
        ccode="""
	    float*param=0;
	    int dim=0;
	    int*dims=0;
	    int *shape=0;
	    int two=1;
	    int count=0;
	    if(0== GetRThread() && 0== GetWThread())
	    {
        """ % d
        for i in xrange(0,self.num):
	    d['x']=inp[i]
	    d['index']=i
	    ccode+="""
	        dim=PyArray_NDIM(%(x)s);
		param = (float*)PyArray_DATA(%(x)s);
		dims=(int*)PyArray_DIMS(%(x)s);
		shape=(int*)PyArray_STRIDES(%(x)s);
		if(1 != dim)
		{
		    two=shape[0]/4;
		}
		WriteArray(\"%(ip)s\",%(port)s,\"%(index)s_%(key)s\",
                    dims[0],two,param);
		two=1;//reset
	    """ % d
	ccode+="""
	    }
	    #if 1
	    while(2< GetWThread())	
	    {
	        count++;	
	    }
	    count=0;
	    #endif
	"""  
	return ccode

class DownParam(gof.Op):
    def __init__(self,num,key=''):
	self.key=key
	self.num=num
    def __hash__(self):
        return hash(type(self)) ^ hash(self.key) ^ hash(self.num)
    def __eq__(self,other):
        return (type(self) == type(other) and
                self.key == other.key and
	        self.num == other.num)
    def make_node(self, x):
        output_var=[]
        for item in x:
            var=item.type()
	    output_var.append(var)
	return gof.Apply(self,x,output_var)
    def c_code_cache_version(self):
        return (1,0)
    def __str__(self):
        return "DownParam(%s,%d)"%(self.key,self.num)
    def c_headers(self):
        return ['PcsParameter.hpp','time.h']
    def c_libraries(self):
        return ['PcsParameter','hiredis','pthread']
    def c_compile_args(self):
        return ['-O0']
    def c_no_compile_args(self):
        return ['-O3','-fopenmp','-O2','-O3']

    def c_code(self,node,name,inp,out,sub):
        fail=sub['fail']
	d={}
	if not config.ipxx or not config.port:
	    print "There is not configure paramter server Ip proxy and port\n"
	    sys.exit(0)
        d['ip']=config.ipxx
	d['port']=config.port
	d['key']=self.key
        d['fail']=fail
	d['num']=self.num
        ccode="""
	    float*output=0;
	    int dim=0;
	    int*dims=0;
	    int *shape=0;
	    int two=1;
	    if(0==GetWThread()&& 0==GetRThread())
	    {
	""" % d
	for i in xrange(0,self.num):
	    d['x']=inp[i]
	    d['z']=out[i]
	    d['index']=i
	    ccode+="""
	        dim=PyArray_NDIM(%(x)s);
	        dims=(int*)PyArray_DIMS(%(x)s);
	        if(0 == %(z)s)
        	{
                    PyArray_Dims outshape;
                    outshape.ptr=PyArray_DIMS(%(x)s);
                    outshape.len=PyArray_NDIM(%(x)s);
                    %(z)s = (PyArrayObject*)PyArray_Newshape(
                            (PyArrayObject*)%(x)s,&outshape,NPY_CORDER);
                }
	        output=(float*)PyArray_DATA(%(z)s);
	        shape=(int*)PyArray_STRIDES(%(x)s);
		if(1 !=dim)
		{
		    two=shape[0]/4;
		}
		ReadArray(\"%(ip)s\",%(port)s,\"%(index)s_%(key)s\",
                        dims[0],two,output);
	        two=1;//reset
	    """ % d
        ccode+="""
            }
	    while(0<GetRThread())
	    {
	        ;	
	    }
	"""
	return ccode


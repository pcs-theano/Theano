""" Ops for downsampling images.

Planned:
DownsampleFactorMax, DownsampleAvg, DownsampleSoftmax.

"""
#This file should move along with conv.py
import __builtin__

import numpy
import sys
import theano
from theano import gof, Op, tensor, Variable, Apply

class NormAcrossMap(Op):
    """
	Local response normalization layer (across maps)
	This layer is just like the one described above, but units are divided
        only by the activities of other units in the same position but in 
        different maps (channels). 
        F[c][x,y]= (1+ a*1.0/Size * sum(F[c- Size/2][x,y]**2, 
                    F[c+Size/2][x,y]**2)) **b
	 
    """
    def __init__(self, a=1e-4, b=0.75, N=5):
        self.a=a
        self.b=b
        self.size=N

    def __eq__(self,other):
        return ( type(self) == type(other) and
		self.size ==other.size and
		self.a == other.a and 
		self.b == other.b)
    def __hash__(self):
        return hash(type(self)) ^ hash(self.size) ^ hash(self.a) ^ hash(self.b)
    def __str__(self):
        return '%s{%s,%s,%s}' %(self.__class__.__name__,
				self.size,self.a,self.b)
    def make_node(self, x):
        if x.type.ndim !=4:
            raise TypeError()
        return gof.Apply(self,[x],[x.type(),x.type()])

    def perform(self,node,inp,out):
        """
	  Op implement in Python
	"""
        x,=inp
        z,scale_data,=out
        if len(x.shape) !=4:
            raise NotImplementedError('NormCrossMap requires 4D input for now')
        if (z[0] is None) or (z[0].shape != z_shape):
            num = reduce(lambda x,y:x*y,x.shape)
            data = numpy.zeros(num).reshape(x.shape)
            z[0] = theano._asarray(data, dtype=x.dtype)
        zz=z[0]
        """zz needs to be initialized with -inf for the following to work """
        zz -=numpy.inf
        #print 'scale_data ',scale_data
        if scale_data[0] ==None:
            num = reduce(lambda x,y:x*y,x.shape)
            o = numpy.zeros(num).reshape(x.shape)
            scale_data[0]=theano._asarray(o,dtype=numpy.float32)
        scale=scale_data[0]
        a=self.a
        b=self.b
        N=self.size
        x_usable2 = x.shape[2]
        x_usable3 = x.shape[3]
        for m in xrange(0,x.shape[0]):
            for c in xrange(0,x.shape[1]):
                c_start = c- (N-1)/2
                c_end =  c_start + N
                if c_start <0:
	            c_start =0
		if c_end > x.shape[1]:
		    c_end = x.shape[1]
		for h in xrange(x_usable2):
		    for w in xrange(x_usable3):
		        scale[m,c,h,w] = 1.0
			for i in xrange(c_start,c_end):
		            value=x[m,i,h,w]
			    scale[m,c,h,w] += ((value *value *a)/N)
		        zz[m,c,h,w]=x[m,c,h,w]/(scale[m,c,h,w] **b)

    def grad(self, inp, grads):
        x,=inp ##input data
        gz,y,=grads ##top_diff
        Out=self(x)##out
        return [NormAcrossMapGrad(self.size,self.a,self.b)(x,Out[0],Out[1],gz)]

    def c_headers(self):
        return ['<math.h>','<mkl.h>']

    def c_libraries(self):
        return ['mkl_rt']

    def c_code(self, node, name, inp, out, sub):
        x,=inp
        z,scale,=out
        fail =sub['fail']
        a=self.a
        b=self.b
        size=self.size
        d={}
        d["x"]=x
        d["z"]=z
        d["a"]=a
        d["b"]=b
        d["size"]=size
        d["scale"]=scale
        ret ="""
        {
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int x_shp0_usable;
        int x_shp1_usable;
        int z_shp0, z_shp1;
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
	x_shp0_usable = PyArray_DIMS(%(x)s)[2];
	x_shp1_usable = PyArray_DIMS(%(x)s)[3];
	z_shp0 = x_shp0_usable;
	z_shp1 = x_shp1_usable;
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1]))
        {
          if(%(z)s)Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_shp0;
          dims[3]=z_shp1;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
        if ((!%(scale)s)
          || *PyArray_DIMS(%(scale)s)!=4
          ||(PyArray_DIMS(%(scale)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(scale)s)[1] != PyArray_DIMS(%(x)s)[1]))
        {
          if(%(scale)s)Py_XDECREF(%(scale)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_shp0;
          dims[3]=z_shp1;
          //TODO: zeros not necessary
          %(scale)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }

        npy_intp num = PyArray_DIMS(%(x)s)[0];
        npy_intp stack = PyArray_DIMS(%(x)s)[1];
        npy_intp elemSize = PyArray_STRIDES(%(x)s)[3];
        dtype_%(scale)s * sc = (dtype_%(scale)s *)PyArray_DATA(%(scale)s);

        #pragma ivdep
        for(int i= 0;i<num* stack * x_shp0_usable * x_shp1_usable;i++)
                sc[i]=1.0;

        dtype_%(x)s * padd_data = (dtype_%(x)s*)malloc((stack+ %(size)s-1)*
                                                        x_shp0_usable*
                                                        x_shp1_usable* 
                                                        elemSize);

        memset(padd_data, 0, (stack+ %(size)s-1)*
                              x_shp0_usable*
                              x_shp1_usable* 
                              elemSize);

        dtype_%(x)s *input = (dtype_%(x)s *)PyArray_DATA(%(x)s);

        dtype_%(z)s * out = (dtype_%(z)s*)PyArray_DATA(%(z)s);

        for(int n =0;n< num;n++)
        {
            vsMul(stack*x_shp0_usable*x_shp1_usable,
                  input+n*stack*x_shp0_usable*x_shp1_usable,
                  input+n*stack*x_shp0_usable*x_shp1_usable,
                  padd_data+ ((%(size)s-1)>>1) * x_shp0_usable * x_shp1_usable);

            for(int c=0;c< %(size)s;c++)//first map
            {
                cblas_saxpy(x_shp0_usable*x_shp1_usable, 
                            1.0 * %(a)s/%(size)s, 
                            padd_data + c* x_shp0_usable * x_shp1_usable, 
                            1, 
                            sc+n*stack*x_shp0_usable*x_shp1_usable, 
                            1);
            }
            for(int c=1;c<stack;c++)
            {
                cblas_scopy(x_shp0_usable*x_shp1_usable, 
                            sc+n*stack*x_shp0_usable*x_shp1_usable + 
                            (c-1)*x_shp0_usable*x_shp1_usable, 
                            1, 
                            sc+(n*stack+c)*x_shp0_usable*x_shp1_usable,
                            1);
                cblas_saxpy(x_shp0_usable*x_shp1_usable, 
                            1.0 * %(a)s/%(size)s, 
                            padd_data + (c + %(size)s-1)* 
                            x_shp0_usable * x_shp1_usable,
                            1,
                            sc+(n*stack+c) * x_shp0_usable * x_shp1_usable, 
                            1);
                cblas_saxpy(x_shp0_usable*x_shp1_usable,
                            -1.0 * %(a)s/%(size)s, 
                            padd_data + (c-1)* x_shp0_usable * x_shp1_usable, 
                            1, 
                            sc+(n*stack+c)*x_shp0_usable*x_shp1_usable,
                            1);
            }
        }
        vsPowx(num*stack*x_shp0_usable*x_shp1_usable,sc,-%(b)s,out);
        vsMul(num * stack * x_shp0_usable * x_shp1_usable,out,input,out);
        free(padd_data);
 	padd_data=0;
    }
	""" % locals()
	return ret
    def c_code_cache_version(self):
        return (0, 1)


class NormAcrossMapGrad(Op):
    """
        Out=gz* f(x)
        f(x) = 1/(1+ (a/N)*sum(x*x))**b-x*2*a*b*sum(x)/(1+(a/N)*sum(x*x))**(b+1)
        Grad Function		
    """
    def __init__(self, N,a,b):
	self.size=N
	self.a=a
	self.b=b

    def __eq__(self, other):
	return (type(self) == type(other) and
		self.size == other.size and 
		self.a == self.a and 
		self.b == self.b)

    def __hash__(self):
	return hash(type(self))^ hash(self.a) ^ hash(self.b) ^ hash(self.size)
	return value

    def __str__(self):
	return '%s{%s,%s,%s}' %(self.__class__.__name__,
				self.size,self.a,self.b)
    def c_headers(self):
        return ['<math.h>','<mkl.h>']

    def c_libraries(self):
        return ['mkl_rt']

    def make_node(self, x, LrnOut,scale, gz):
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(LrnOut, Variable) and LrnOut.ndim == 4
	assert isinstance(scale, Variable) and scale.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        return gof.Apply(self,[x,LrnOut,scale,gz],[x.type()])


    def perform(self, node, inp, out):
        """
          Op implement in Python
        """
        x,LrnOut,scale,gz,=inp
        z,=out
        if len(x.shape) !=4:
            raise NotImplementedError('NormCrossMap requires 4D input for now')

        if (z[0] is None) or (z[0].shape != z_shape):
            num = reduce(lambda x,y:x*y,x.shape)
            data = numpy.zeros(num).reshape(x.shape)
            z[0] = theano._asarray(data, dtype=x.dtype)
        zz=z[0]
        """zz needs to be initialized with -inf for the following to work """
        zz -=numpy.inf
        a=self.a
        b=self.b
        N=self.size
        x_usable2 = x.shape[2]
        x_usable3 = x.shape[3]
	print 'grads: ', LrnOut[0,0,0,0],LrnOut[0,1,0,0] ,LrnOut[0,2,0,0]
	print 'scale ',scale[0,0,0,0],scale[0,1,0,0],scale[0,2,0,0]
	##undo Perform implement
        for m in xrange(0,x.shape[0]):
            for c in xrange(0,x.shape[1]):
	        c_start = c- (N-1)/2
                c_end =  c_start + N
		if c_start <0:
	            c_start =0
		if c_end > x.shape[1]:
		    c_end =x.shape[1]
                for h in xrange(x_usable2):
                    for w in xrange(x_usable3):
                        value=0.0
                        for i in xrange(c_start,c_end):
                            value +=((LrnOut[m,i,h,w]/scale[m,i,h,w])*
                                     (2.0*a*b/N))
		            if m== 0 and h ==0 and w== 0 and c==0:
			        print (c_end,value,LrnOut[m,i,h,w],
                                       scale[m,i,h,w], 
                                       LrnOut[m,i,h,w]*2.0*a*b)
		        if m==0 and c ==0 and h ==0 and w==0:
			    print ('complete ', value,
                                   x[m,c,h,w],scale[m,c,h,w])
			    print ((scale[m,c,h,w]**-b), ' - ',
                                    value*x[m,c,h,w]*gz[m,c,h,w], '=',
                                    ((scale[m,c,h,w]**(-b)) - value*x[m,c,h,w])*
                                    gz[m,c,h,w])
                        value= (scale[m,c,h,w] ** (-b)) - value * x[m,c,h,w] 
                        zz[m,c,h,w]=value*gz[m,c,h,w]

    def c_code(self, node, name, inp, out, sub):
	x,LrnOut,scale,gz,=inp
	z,=out
        a=self.a
	b=self.b
	size=self.size
	d={}
	d["a"]=a
	d["x"]=x
	d["z"]=z
	d["gz"]=gz
	d["LrnOut"]=LrnOut
	d["scale"]=scale
	d["size"]=size
	fail=sub['fail']
	ret = """
        { 
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int x_shp0_usable;
        int x_shp1_usable;
        int z_shp0, z_shp1;
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        x_shp0_usable = PyArray_DIMS(%(x)s)[2];
        x_shp1_usable = PyArray_DIMS(%(x)s)[3];
        z_shp0 = x_shp0_usable;
        z_shp1 =  x_shp1_usable;
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          )
        {
          if(%(z)s)Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_shp0;
          dims[3]=z_shp1;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
	npy_intp num=PyArray_DIMS(%(x)s)[0];
	npy_intp channels=PyArray_DIMS(%(x)s)[1];
	/*padd*/
        npy_intp dims[4] = {0,0,0,0};
        dims[0]=num;
        dims[1]= channels + %(size)s -1;
        dims[2]=x_shp0_usable;
        dims[3]=x_shp1_usable;
	npy_intp elemSize= PyArray_STRIDES(%(x)s)[3];
       	dtype_%(x)s * padd_data = (dtype_%(x)s*)malloc(dims[0]*dims[1]*dims[2]*
                                                       dims[3]*elemSize);
	memset(padd_data,0,dims[0]*dims[1]*dims[2]*dims[3]*elemSize);
	dims[0]=1;
	dims[1]=1;
	dtype_%(x)s * accum_data = (dtype_%(x)s*)malloc(dims[0]*dims[1]*dims[2]*
                                                        dims[3]*elemSize);
	dtype_%(x)s * accum_bottom = (dtype_%(x)s*)malloc(dims[0]*dims[1]*
                                                          dims[2]*dims[3]*
                                                          elemSize);
	

	dtype_%(scale)s * scale_data =
            (dtype_%(scale)s *)PyArray_GETPTR4(%(scale)s, 0,0,0,0); //scale
	dtype_%(gz)s * gz_ptr =
            (dtype_%(gz)s *)PyArray_GETPTR4(%(gz)s, 0,0,0,0); //gz
	dtype_%(z)s * out =
            (dtype_%(z)s *)PyArray_GETPTR4(%(z)s, 0,0,0,0); //OutPut
	dtype_%(LrnOut)s * lrn =
            (dtype_%(LrnOut)s *)PyArray_GETPTR4(%(LrnOut)s, 0,0,0,0); //LrnOut
	dtype_%(x)s * input =
            (dtype_%(x)s *)PyArray_GETPTR4(%(x)s, 0,0,0,0); //input

	vsPowx(num * channels*x_shp0_usable*x_shp1_usable,
               scale_data, -%(b)s, out);
	vsMul(num * channels*x_shp0_usable*x_shp1_usable,
              gz_ptr, out, out);

	int pre_offset = ((%(size)s -1 )>>1)*x_shp0_usable*x_shp1_usable;
	for(int n =0;n<num;n++)
	{
           int block_offset = n*channels*x_shp0_usable*x_shp1_usable;

	   vsMul(channels*x_shp0_usable*x_shp1_usable,
                 gz_ptr+block_offset,lrn+block_offset,padd_data+pre_offset);

	   vsDiv(channels*x_shp0_usable*x_shp1_usable, padd_data+pre_offset,
                 scale_data+block_offset,padd_data+pre_offset);	 
	   
	   memset(accum_data, 0, x_shp0_usable * x_shp1_usable * elemSize);  
	   
	   for(int c=0;c< %(size)s-1;c++)
	   {
               cblas_saxpy(x_shp0_usable*x_shp1_usable, 1.0,
                           padd_data + c* x_shp0_usable * x_shp1_usable,
                           1, accum_data, 1);
	   }
	
	   for(int c=0;c<channels;c++)
	   {
		cblas_saxpy(x_shp0_usable*x_shp1_usable, 1.0,
                padd_data + (c + %(size)s-1)* x_shp0_usable * x_shp1_usable,
                1, accum_data, 1);

		vsMul(x_shp0_usable*x_shp1_usable,
                      input+block_offset+c*x_shp0_usable * x_shp1_usable,
                      accum_data,accum_bottom);

		cblas_saxpy(x_shp0_usable*x_shp1_usable, 
                            -2.0 * %(a)s * %(b)s / %(size)s,
                            accum_bottom, 1, 
                            out +block_offset + c*x_shp0_usable * x_shp1_usable,
                            1);
		
		cblas_saxpy(x_shp0_usable*x_shp1_usable,
                            -1.0,
                            padd_data + c * x_shp0_usable * x_shp1_usable,
                            1,accum_data, 1);
	   }//for(c)
	}//for(num)
	free(accum_data);
	accum_data=0;
	free(accum_bottom);
	accum_bottom=0;
	free(padd_data);
	padd_data=0;
 }
""" % locals()
	return ret

    def c_code_cache_version(self):
        return (0, 1)

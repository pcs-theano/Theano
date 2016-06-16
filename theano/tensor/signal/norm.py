import __builtin__  ##FIXME, do we need this??

import sys
import numpy
import theano
from theano import gof, Op, tensor, Variable, Apply


class NormAcrossMap(Op):
    """
    Local Response Normalization (Across Maps)

    Refer to the below link for the definition of LRN
        https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_
        response_normalization_layer_(across_maps)

    The activity of a neuron is divided only by the "adjacent" of activities
    which are in the same spatial postion but in different maps (channels).
    'c' stands for current channel index.

        F[c][x,y] = (1 + (alpha*1.0/n) * sum(F[c - n/2][x,y]^2,
                    F[c + n/2][x,y]^2))^beta
    
    Parameters
    ----------
    alpha:
    beta :
    n    : indicates how many nearby maps to use for normalization.
    """

    def __init__(self, alpha = 1e-4, beta = 0.75, n = 5):
        self.alpha = alpha
        self.beta = beta
        self.size = n

    def __eq__(self, other):
        return (type(self) == type(other) and
		self.alpha == other.alpha and 
		self.beta == other.beta and
		self.size == other.size)

    def __hash__(self):
        return (hash(type(self)) ^ hash(self.alpha) ^ \
                hash(self.beta) ^ hash(self.size))

    def __str__(self):
        return '%s{%s, %s, %s}' %(self.__class__.__name__,
				self.alpha, self.beta, self.size)

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()

        x = tensor.as_tensor_variable(x)
        return gof.Apply(self, [x], [x.type(), x.type()])

    def perform(self, node, inp, out):
        x, = inp
        z, scale_data, = out

        if len(x.shape) != 4:
            raise NotImplementedError('NormCrossMap requires 4D input')

        if (z[0] is None):
            num = reduce(lambda x, y : x * y, x.shape)
            data = numpy.zeros(num).reshape(x.shape)
            z[0] = theano._asarray(data, dtype = x.dtype)
        zz = z[0]
        """zz needs to be initialized with -inf for the following to work """
        zz -= numpy.inf
        #print 'scale_data ',scale_data
        if (scale_data[0] is None):
            num = reduce(lambda x, y : x * y, x.shape)
            o = numpy.zeros(num).reshape(x.shape)
            scale_data[0] = theano._asarray(o, dtype = numpy.float32)
        scale = scale_data[0]

        alpha = self.alpha
        beta = self.beta
        size = self.size
        x_row = x.shape[2]
        x_col = x.shape[3]

        for bi in xrange(0, x.shape[0]):
            for c in xrange(0, x.shape[1]):
                c_start = c - (size - 1) / 2
                c_end = c_start + size

                if c_start < 0:
	            c_start = 0
		if c_end > x.shape[1]:
		    c_end = x.shape[1]

		for h in xrange(x_row):
		    for w in xrange(x_col):
		        scale[bi, c, h, w] = 1.0
			for i in xrange(c_start, c_end):
		            value = x[bi, i, h, w]
			    scale[bi, c, h, w] += ((value * value * alpha)/size)
		        zz[bi, c, h, w] = \
                            x[bi, c, h, w] / (scale[bi, c, h, w]**beta)

    def grad(self, inp, grads):
        x, = inp
        gz, y, = grads
        out = self(x)
        return [NormAcrossMapGrad(alpha=self.alpha, beta=self.beta,  \
                                  n=self.size)(x, out[0], out[1], gz)]

    def c_headers(self):
        return ['<math.h>','<mkl.h>']  ##FIXME

    def c_libraries(self):
        return ['mkl_rt']  ##FIXME, what if user don't have mkl?

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, scale, = out
        fail = sub['fail']
        alpha = self.alpha
        beta = self.beta
        size = self.size
        d = {}
        d["x"] = x
        d["z"] = z
        d["alpha"] = self.alpha
        d["beta"] = self.beta
        d["size"] = self.size
        d["scale"] = scale
        ret = """
        {
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
	int x_row = PyArray_DIMS(%(x)s)[2];
	int x_col = PyArray_DIMS(%(x)s)[3];
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s) != 4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1]))
        {
          if(%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0, 0, 0, 0};
          dims[0] = PyArray_DIMS(%(x)s)[0];
          dims[1] = PyArray_DIMS(%(x)s)[1];
          dims[2] = x_row;
          dims[3] = x_col;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum, 0);
        }

        if ((!%(scale)s)
          || *PyArray_DIMS(%(scale)s) != 4
          ||(PyArray_DIMS(%(scale)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(scale)s)[1] != PyArray_DIMS(%(x)s)[1]))
        {
          if(%(scale)s) Py_XDECREF(%(scale)s);
          npy_intp dims[4] = {0, 0, 0, 0};
          dims[0] = PyArray_DIMS(%(x)s)[0];
          dims[1] = PyArray_DIMS(%(x)s)[1];
          dims[2] = x_row;
          dims[3] = x_col;
          //TODO: zeros not necessary
          %(scale)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }

        npy_intp x_bs = PyArray_DIMS(%(x)s)[0];
        npy_intp x_channels = PyArray_DIMS(%(x)s)[1];
        npy_intp elemSize = PyArray_STRIDES(%(x)s)[3];
        dtype_%(scale)s * sc = (dtype_%(scale)s *)PyArray_DATA(%(scale)s);

        int x_total_size = x_bs * x_channels * x_row * x_col;
        #pragma ivdep
        for(int i = 0; i < x_total_size; i++)
            sc[i]=1.0;

        dtype_%(x)s *padd_data = (dtype_%(x)s*)malloc(
                (x_channels + %(size)s - 1) * x_row * x_col * elemSize);
        memset(padd_data, 0, (x_channels + %(size)s - 1) * x_row * \
                x_col * elemSize);

        dtype_%(x)s *input = (dtype_%(x)s *)PyArray_DATA(%(x)s);
        dtype_%(z)s *out = (dtype_%(z)s *)PyArray_DATA(%(z)s);

        for(int bi = 0; bi < x_bs; bi++)
        {
            vsMul(x_channels * x_row * x_col, \
                    input + bi * x_channels * x_row * x_col, \
                    input + bi * x_channels * x_row * x_col, \
                    padd_data + ((%(size)s - 1) >> 1) * x_row * x_col);

            for(int c = 0; c < %(size)s; c++)
            {
                cblas_saxpy(x_row * x_col, 1.0 * %(alpha)s/%(size)s, \
                        padd_data + c * x_row * x_col, \
                        1, sc + bi * x_channels * x_row*x_col, 1);
            }
            for(int c = 1; c < x_channels; c++)
            {
                cblas_scopy(x_row * x_col, \
                        sc + bi * x_channels * x_row * x_col + \
                        (c - 1) * x_row * x_col, \
                        1, sc + (bi * x_channels + c) * x_row * x_col, 1);

                cblas_saxpy(x_row * x_col, 1.0 * %(alpha)s/%(size)s, \
                        padd_data + (c + %(size)s - 1) * x_row * x_col, \
                            1, sc + (bi * x_channels + c) * x_row * x_col, 1);
                cblas_saxpy(x_row * x_col, -1.0 * %(alpha)s/%(size)s, \
                            padd_data + (c - 1) * x_row * x_col, \
                            1, sc + (bi * x_channels + c) * x_row * x_col, 1);
            }
        }
        vsPowx(x_bs * x_channels * x_row * x_col, sc, -%(beta)s, out);
        vsMul(x_bs * x_channels * x_row * x_col, out, input, out);
        free(padd_data);
 	padd_data = NULL;
        }
	""" % locals()
	return ret

    def c_code_cache_version(self):
        return (0, 1)


class NormAcrossMapGrad(Op):
    """
    Grad Function of NormAcrossMap		
        roOut = gz * f(x)
        f(x) = 1/(1 + (alpha/n)*sum(x*x))**beta - 2*x*alpha*beta*sum(x)/(1+(alpha/n)*sum(x*x))**(beta+1)

    Parameters
    ----------
    alpha:
    beta :
    n    : indicates how many nearby maps to use for normalization.

    """
    def __init__(self, alpha=None, beta=None, n=None):
	self.alpha = alpha
	self.beta = beta
	self.size = n

    def __eq__(self, other):
	return (type(self) == type(other) and
		self.alpha == self.alpha and 
		self.beta == self.beta and
		self.size == other.size)

    def __hash__(self):
	return (hash(type(self)) ^ hash(self.alpha) ^ \
                hash(self.beta) ^ hash(self.size))

    def __str__(self):
	return '%s{%s,%s,%s}' %(self.__class__.__name__,
				self.alpha, self.beta, self.size)

    def c_headers(self):
        return ['<math.h>','<mkl.h>'] ##FIXME

    def c_libraries(self):
        return ['mkl_rt'] ##FIXME

    def make_node(self, x, LrnOut, scale, gz):
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(LrnOut, Variable) and LrnOut.ndim == 4
	assert isinstance(scale, Variable) and scale.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        return gof.Apply(self, [x, LrnOut, scale, gz], [x.type()])

    def perform(self, node, inp, out):
        x, LrnOut, scale, gz, = inp
        z, = out

        if len(x.shape) != 4:
            raise NotImplementedError('NormCrossMap requires 4D input for now')

        if (z[0] is None):
            num = reduce(lambda x, y : x * y, x.shape)
            data = numpy.zeros(num).reshape(x.shape)
            z[0] = theano._asarray(data, dtype=x.dtype)
        zz = z[0]
        """zz needs to be initialized with -inf for the following to work """
        zz -= numpy.inf
        alpha = self.alpha
        beta = self.beta
        size = self.size
        x_row = x.shape[2]
        x_col = x.shape[3]
        ## FIXME, after debug, just remove
	#print 'grads: ', LrnOut[0,0,0,0],LrnOut[0,1,0,0] ,LrnOut[0,2,0,0]
	#print 'scale ',scale[0,0,0,0],scale[0,1,0,0],scale[0,2,0,0]

        for bi in xrange(0, x.shape[0]):
            for c in xrange(0, x.shape[1]):
	        c_start = c - (size - 1)/2
                c_end =  c_start + size
		if c_start < 0:
	            c_start = 0
		if c_end > x.shape[1]:
		    c_end = x.shape[1]
                for h in xrange(x_row):
                    for w in xrange(x_col):
                        value = 0.0
                        for i in xrange(c_start, c_end):
                            value += ((LrnOut[bi, i, h, w]/scale[bi, i, h, w]) *
                                     (2.0 * alpha * beta / size))
		            #if bi == 0 and h == 0 and w == 0 and c == 0: ##FIXME, after debug, just remove it
			    #    print (c_end,value,LrnOut[bi,i,h,w],
                            #           scale[bi,i,h,w], 
                            #           LrnOut[bi,i,h,w]*2.0*alpha*beta)
		        #if bi==0 and c ==0 and h ==0 and w==0: ##FIXME
			#    print ('cobiplete ', value,
                        #           x[bi,c,h,w],scale[bi,c,h,w])
			#    print ((scale[bi,c,h,w]**-beta), ' - ',
                        #            value*x[bi,c,h,w]*gz[bi,c,h,w], '=',
                        #            ((scale[bi,c,h,w]**(-beta)) - value*x[bi,c,h,w])*
                        #            gz[bi,c,h,w])
                        value = (scale[bi, c, h, w] ** (-beta)) - \
                                value * x[bi, c, h, w] 
                        zz[bi, c, h, w] = value * gz[bi, c, h, w]

    def c_code(self, node, name, inp, out, sub):
	x, LrnOut, scale, gz, =inp
	z, = out
        alpha = self.alpha
	beta = self.beta
	size = self.size
	d={}
	d["alpha"] = alpha
	d["x"] = x
	d["z"] = z
	d["gz"] = gz
	d["LrnOut"] = LrnOut
	d["scale"] = scale
	d["size"] = size
	fail = sub['fail']
	ret = """
        { 
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        if(PyArray_NDIM(%(x)s) != 4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        int x_row = PyArray_DIMS(%(x)s)[2];
        int x_col = PyArray_DIMS(%(x)s)[3];
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          )
        {
          if(%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0, 0, 0, 0};
          dims[0] = PyArray_DIMS(%(x)s)[0];
          dims[1] = PyArray_DIMS(%(x)s)[1];
          dims[2] = x_row;
          dims[3] = x_col;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum, 0);
        }
	npy_intp x_bs = PyArray_DIMS(%(x)s)[0];
	npy_intp x_channels = PyArray_DIMS(%(x)s)[1];
	npy_intp elemSize = PyArray_STRIDES(%(x)s)[3];

	/*padd*/
        npy_intp dims[4] = {0, 0, 0, 0};
        dims[0] = x_bs;
        dims[1] = x_channels + %(size)s -1;
        dims[2] = x_row;
        dims[3] = x_col;

        npy_intp total_size = dims[0] * dims[1] * dims[2] * dims[3] * elemSize;
       	dtype_%(x)s *padd_data = (dtype_%(x)s*)malloc(total_size);
	memset(padd_data, 0, total_size);

        dims[0] = 1;
	dims[1] = 1;
        total_size = dims[0] * dims[1] * dims[2] * dims[3] * elemSize;
	dtype_%(x)s *accum_data = (dtype_%(x)s*)malloc(total_size);
	dtype_%(x)s *accum_bottom = (dtype_%(x)s*)malloc(total_size);

	dtype_%(scale)s *scale_data =
            (dtype_%(scale)s *)PyArray_GETPTR4(%(scale)s, 0, 0, 0, 0); //scale
	dtype_%(gz)s *gz_ptr =
            (dtype_%(gz)s *)PyArray_GETPTR4(%(gz)s, 0, 0, 0, 0); //gz
	dtype_%(z)s *out =
            (dtype_%(z)s *)PyArray_GETPTR4(%(z)s, 0, 0, 0, 0); //outPut
	dtype_%(LrnOut)s *lrn =
            (dtype_%(LrnOut)s *)PyArray_GETPTR4(%(LrnOut)s, 0, 0, 0, 0); //LrnOut
	dtype_%(x)s *input =
            (dtype_%(x)s *)PyArray_GETPTR4(%(x)s, 0, 0, 0, 0); //input

	vsPowx(x_bs * x_channels * x_row * x_col, scale_data, -%(beta)s, out);
	vsMul(x_bs * x_channels * x_row * x_col, gz_ptr, out, out);

	int pre_offset = ((%(size)s - 1) >> 1) * x_row * x_col;
	for(int n = 0; n < x_bs; n++)
	{
           int block_offset = n * x_channels * x_row * x_col;

	   vsMul(x_channels * x_row * x_col, gz_ptr + block_offset, \
                   lrn + block_offset, padd_data + pre_offset);

	   vsDiv(x_channels * x_row * x_col, padd_data + pre_offset, \
                 scale_data + block_offset, padd_data + pre_offset);	 
	   
	   memset(accum_data, 0, x_row * x_col * elemSize);  
	   
	   for(int c = 0; c < %(size)s - 1; c++)
	   {
               cblas_saxpy(x_row * x_col, 1.0, \
                       padd_data + c* x_row * x_col, 1, accum_data, 1);
	   }
	
	   for(int c = 0; c < x_channels; c++)
	   {
		cblas_saxpy(x_row * x_col, 1.0, \
                        padd_data + (c + %(size)s - 1) * x_row * x_col, \
                        1, accum_data, 1);

		vsMul(x_row * x_col, input + block_offset + c * x_row * x_col,\
                        accum_data, accum_bottom);

		cblas_saxpy(x_row * x_col, \
                        -2.0 * %(alpha)s * %(beta)s / %(size)s, \
                        accum_bottom, 1, \
                        out + block_offset + c * x_row * x_col, 1);
		
		cblas_saxpy(x_row * x_col, -1.0, \
                        padd_data + c * x_row * x_col, 1, accum_data, 1);
	   }//for(c)
	}//for(x_bs)
	free(accum_data);
	accum_data = NULL;
	free(accum_bottom);
	accum_bottom = NULL;
	free(padd_data);
	padd_data = NULL;
        }
        """ % locals()
	return ret

    def c_code_cache_version(self):
        return (0, 1)

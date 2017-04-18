import numpy

try:
    import theano
    import mkl_ndarray
    from mkl_ndarray import mkl_ndarray as mnda
except Exception as e:
    raise e


a = numpy.random.rand(2, 3, 4, 5).astype(numpy.float32)
b = mnda.MKLNdarray(a)
c = b.__array__(b)
d = numpy.asarray(b)

assert numpy.allclose(a, c)
assert numpy.allclose(a, b)
assert b.shape == (2 ,3 , 4, 5)
assert b.size == 2 * 3 * 4 * 5
assert b.dtype == 'float32'
assert b.ndim == 4


print('test_mkl_ndarray.py pass...')

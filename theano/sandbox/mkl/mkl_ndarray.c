#include <Python.h>
#include <structmember.h>
#include "theano_mod_helper.h"
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "mkl_ndarray.h"


int MKLNdarray_Check(const PyObject *ob);
PyObject* MKLNdarray_New(int nd, int typenum);
int MKLNdarray_CopyFromArray(MKLNdarray *self, PyArrayObject *obj);


/*
 * This function is called in MKLNdarray_dealloc.
 *
 * Release all allocated buffer and layout in self.
 */
static int
MKLNdarray_uninit(MKLNdarray *self) {

    printf("MKLNdarray_uninit %p \n", self);
    int rval = 0;

    if (self->dtype == MKL_FLOAT64) {  // for float64
        if (self->private_data) {
            rval = dnnReleaseBuffer_F64(self->private_data);

            if (rval != 0) {
                PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_uninit: fail to release data \n");
            }

            self->private_data = NULL;
        }

        self->data_size = 0;
        if (self->private_layout) {
            rval = dnnLayoutDelete_F64(self->private_layout);

            if (rval != 0) {
                PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_uninit: fail to release layout \n");
            }

            self->private_layout = NULL;
        }

        if (self->private_workspace) {
            rval = dnnReleaseBuffer_F64(self->private_workspace);

            if (rval != 0) {
                PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_uninit: fail to release workspace \n");
            }

            self->private_workspace = NULL;
        }
    } else {  // for float32
        if (self->private_data) {
            rval = dnnReleaseBuffer_F32(self->private_data);

            if (rval != 0) {
                PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_uninit: fail to release data \n");
            }

            self->private_data = NULL;
        }

        self->data_size = 0;
        if (self->private_layout) {
            rval = dnnLayoutDelete_F32(self->private_layout);

            if (rval != 0) {
                PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_uninit: fail to release layout \n");
            }

            self->private_layout = NULL;
        }

        if (self->private_workspace) {
            rval = dnnReleaseBuffer_F32(self->private_workspace);

            if (rval != 0) {
                PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_uninit: fail to release workspace \n");
            }

            self->private_workspace = NULL;
        }
    }

    self->nd = -1;
    self->dtype = 11;

    Py_XDECREF(self->base);
    self->base = NULL;

    return rval;
}


/*
 * type:tp_dealloc
 *
 * This function will be called by Py_DECREF when object's reference count is reduced to zero.
 * DON'T call this function directly.
 *
 */
static void
MKLNdarray_dealloc(MKLNdarray *self) {

    printf("MKLNdarray_dealloc\n");
    if (Py_REFCNT(self) > 1) {
        printf("WARNING: MKLNdarray_dealloc called when there is still active reference to it.\n");
    }

    MKLNdarray_uninit(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


/*
 * type:tp_new
 *
 * This function is used to create an instance of object.
 * Be first called when do a = MKLNdarray() in python code.
 *
 */
static PyObject*
MKLNdarray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

    MKLNdarray* self = NULL;
    printf("MKLNdarray_new\n");
    self = (MKLNdarray*)(type->tp_alloc(type, 0));

    if (self != NULL) {
        self->base              = NULL;
        self->nd                = -1;
        self->dtype             = -1;
        self->private_workspace = NULL;
        self->private_data      = NULL;
        self->private_layout    = NULL;
        self->data_size         = 0;

        memset((void*)(self->user_structure), 0, 2 * MAX_NDIM * sizeof (size_t));
    } else {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_new: fail to create a new instance \n");
        return NULL;
    }
    return (PyObject*)self;
}



/*
 * type:tp_init
 *
 * This function is called after MKLNdarray_new.
 *
 * Initialize an instance. like __init__() in python code.
 *
 * args: need input a PyArrayObject here.
 *
 */
static int
MKLNdarray_init(MKLNdarray *self, PyObject *args, PyObject *kwds) {

    printf("MKLNdarray_init\n");
    PyObject* arr = NULL;

    if (!PyArg_ParseTuple(args, "O", &arr))
        return -1;

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "PyArrayObject arg required");
        return -1;
    }

    // do type conversion here. PyArrayObject -> MKLNdarray
    int rval = MKLNdarray_CopyFromArray(self, (PyArrayObject*)arr);
    return rval;
}


/*
 * type:tp_repr
 *
 * Return a string or a unicode object. like repr() in python code.
 *
 */
PyObject* MKLNdarray_repr(PyObject *self) {

    MKLNdarray* object = (MKLNdarray*)self;
    char cstr[64]; // 64 chars is enough for a string.

    sprintf(cstr, "ndim=%d, dtype=%s", object->nd, MKL_TYPE[object->dtype]);

    PyObject* out = PyString_FromFormat("%s%s%s", "MKLNdarray(", cstr, ")");
#if PY_MAJOR_VERSION >= 3
    PyObject* out2 = PyObject_Str(out);
    Py_DECREF(out);
    return out2;
#endif
    return out;
}


/*
 * Get dims in user_structure.
 *
 * A pointer is returned.
 *
 */
const size_t*
MKLNdarray_DIMS(const MKLNdarray *self) {
    return self->user_structure;
}


/*
 * Get strides in user_structure.
 *
 * A pointer is returned. stride has a self->nd offset n user_structure
 *
 */
const size_t*
MKLNdarray_STRIDES(const MKLNdarray *self) {
    return self->user_structure + self->nd;
}


/*
 * Get ndim.
 *
 * An integer is returned.
 *
 */
int MKLNdarray_NDIM(const MKLNdarray *self) {
    return self->nd;
}


/*
 * Get dtype.
 *
 * An integer is returned.
 *
 */
int MKLNdarray_TYPE(const MKLNdarray *self) {
    return self->dtype;
}


/*
 * Get address of private_data.
 *
 * An void* pointer is returned.
 *
 */
void*
MKLNdarray_DATA(const MKLNdarray *self) {
    return self->private_data;
}


/*
 * Get address of private_workspace.
 *
 * An void* pointer is returned.
 *
 */
void*
MKLNdarray_WORKSPACE(const MKLNdarray *self) {
    return self->private_workspace;
}


/*
 * Get address of private_layout.
 *
 * An dnnLayout_t* pointer is returned.
 *
 */
dnnLayout_t*
MKLNdarray_LAYOUT(const MKLNdarray *self) {
    return (dnnLayout_t*)(&(self->private_layout));
}


/*
 * In this function a plain layout is created for self.
 *
 * A private_data buffer is allocated for self according to the plain layout.
 *
 */
static int
MKLNdarray_allocate_mkl_buffer(MKLNdarray *self) {

    if (self->nd <= 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "Can't create mkl dnn layout and allocate buffer for a %d dimension MKLNdarray",
                     self->nd);
        return -1;
    }
    size_t ndim = self->nd;

    if (self->private_layout || self->private_data) {
        PyErr_Format(PyExc_RuntimeError,
                     "MKL layout and buffer have been allocated for %p \n", self);
        return -1;
    }

    // float64
    if (self->dtype == MKL_FLOAT64) {
        int status = dnnLayoutCreate_F64(&(self->private_layout),
                                         ndim,
                                         self->user_structure,
                                         self->user_structure + self->nd);
        if (0 != status || NULL == self->private_layout) {
            PyErr_Format(PyExc_RuntimeError,
                         "Call dnnLayoutCreate_F64 failed: %d",
                         status);
            return -1;
        }

        status = dnnAllocateBuffer_F64(&(self->private_data), self->private_layout);
        if (0 != status || NULL == self->private_data) {
            PyErr_Format(PyExc_RuntimeError,
                         "Call dnnAllocateBuffer_F64 failed: %d",
                         status);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F64(self->private_layout);

    } else {  // float32
        int status = dnnLayoutCreate_F32(&(self->private_layout),
                                                 ndim,
                                                 self->user_structure,
                                                 self->user_structure + self->nd);
        if (0 != status || NULL == self->private_layout) {
            PyErr_Format(PyExc_RuntimeError,
                         "Call dnnLayoutCreate_F32 failed: %d",
                         status);
            return -1;
        }

        status = dnnAllocateBuffer_F32(&(self->private_data), self->private_layout);
        if (0 != status || NULL == self->private_data) {
            PyErr_Format(PyExc_RuntimeError,
                         "Call dnnAllocateBuffer_F32 failed: %d",
                         status);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F32(self->private_layout);
    }

    return 0;
}


/*
 * Set user_structure for self.
 *
 * nd: number of dimension. nd should <= 16.
 *
 * dims: dimension info
 *
 */
int MKLNdarray_set_structure(MKLNdarray *self, int nd, const size_t *dims) {

    assert (self->nd == nd);

    if (nd > MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray does not support a %d-dim array. Try array which ndim is <= %d", nd, MAX_NDIM);
        return -1;
    }

    self->user_structure[0] = dims[0];
    self->user_structure[2 * nd - 1] = 1;

    for (int i = 1; i < nd; i++) {
        // nchw
        self->user_structure[i] = dims[i];
        // chw, hw, w, 1
        self->user_structure[2 * nd - 1 - i] = self->user_structure[2 * nd - i] * dims[nd - i];
    }

    return 0;
}



/*
 * Copy/construct a plain MKLNdarray with dada/structure from a PyArrayObject.
 *
 * Need check the dtype and ndim of PyArrayObject.
 *
 */
int MKLNdarray_CopyFromArray(MKLNdarray *self, PyArrayObject *obj) {
    int ndim = PyArray_NDIM(obj);
    npy_intp* d = PyArray_DIMS(obj);
    int typenum = PyArray_TYPE(obj);

    if (typenum != MKL_FLOAT32 && typenum != MKL_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "MKLNdarray_CopyFromArray: can only copy from float/double arrays");
        return -1;
    }

    if (ndim < 0 || ndim > MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray does not support a %d-dim array. Try array which ndim is <= %d", ndim, MAX_NDIM);
        return -1;
    }

    self->dtype = typenum;
    self->nd = ndim;
    size_t dims[MAX_NDIM] = {0};
    size_t user_size = 1;

    for (int i = 0; i < ndim; i++) {
        dims[i] = (size_t)d[i];
        user_size *= dims[i];
    }

    int err = MKLNdarray_set_structure(self, ndim, dims);
    if (err < 0) {
        return err;
    }

    // prepare user layout and mkl buffer
    err = MKLNdarray_allocate_mkl_buffer(self);
    if (err < 0) {
        return err;
    }

    // copy data to mkl buffer
    size_t element_size = (size_t)PyArray_ITEMSIZE(obj);
    // assert (user_size * element_size <= self->data_size);
    memcpy((void*)self->private_data, (void*)PyArray_DATA(obj), user_size * element_size);
    return 0;
}



/*
 * Create a MKLNdarray object according to input dims and typenum.
 *
 * Set all elements to zero.
 *
 * n: number of dimension
 * dims: dimension info
 * typenum: 11 means float32, 12 means float64.
 *
 */
PyObject* MKLNdarray_ZEROS(int n, size_t *dims, int typenum) {

    size_t total_elements = 1;

    if (n < 0 || n > MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray does not support a %d-dim array. Try array which ndim is <= %d", n, MAX_NDIM);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        if (dims[i] != 0 && total_elements > (SIZE_MAX / dims[i])) {
            PyErr_Format(PyExc_RuntimeError,
                         "Can't store in size_t for the bytes requested %llu * %llu",
                         (unsigned long long)total_elements,
                         (unsigned long long)dims[i]);
            return NULL;
        }
        total_elements *= dims[i];
    }

    // total_elements now contains the size of the array
    size_t max = 0;
    if (typenum == MKL_FLOAT64)
        max = SIZE_MAX / sizeof (double);
    else
        max = SIZE_MAX / sizeof (float);

    if (total_elements > max) {
        PyErr_Format(PyExc_RuntimeError,
                     "Can't store in size_t for the bytes requested %llu",
                     (unsigned long long)total_elements);
        return NULL;
    }

    size_t total_size = 0;
    if (typenum == MKL_FLOAT64)
        total_size = total_elements * sizeof (double);
    else
        total_size = total_elements * sizeof (float);

    MKLNdarray* rval = (MKLNdarray*)MKLNdarray_New(n, typenum);
    if (!rval) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_ZEROS: call to New failed");
        return NULL;
    }

    if (MKLNdarray_set_structure(rval, n, dims)) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_ZEROS: syncing structure to mkl failed.");
        Py_DECREF(rval);
        return NULL;
    }

    if (MKLNdarray_allocate_mkl_buffer(rval)) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarrya_ZEROS: allocation failed.");
        Py_DECREF(rval);
        return NULL;
    }
    // Fill with zeros
    memset(rval->private_data, 0, total_size);

    return (PyObject*)rval;
}


/*
 * Get shape info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a tuple contains dimension info.
 */
static PyObject*
MKLNdarray_get_shape(MKLNdarray *self, void *closure) {

    if (self->nd < 0 || self->dtype < 0) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray not initialized");
        return NULL;
    }

    PyObject* rval = PyTuple_New(self->nd);
    if (rval == NULL) {
        return NULL;
    }

    for (int i = 0; i < self->nd; i++) {
        if (PyTuple_SetItem(rval, i, PyInt_FromLong(MKLNdarray_DIMS(self)[i]))) {
            Py_XDECREF(rval);
            return NULL;
        }
    }

    return rval;
}


/*
 * Get dtype info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a string: 'float32' or 'float64'.
 *
 */
static PyObject*
MKLNdarray_get_dtype(MKLNdarray *self, void *closure) {

    if (self->nd < 0 || self->dtype < 0) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray not initialized");
        return NULL;
    }

    PyObject * rval = PyString_FromFormat("%s", MKL_TYPE[self->dtype]);
    return rval;
}


/*
 * Get ndim info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a integer number. If self is not initialized, -1 will be returned.
 *
 */
static PyObject*
MKLNdarray_get_ndim(MKLNdarray *self, void *closure) {
    return PyInt_FromLong(self->nd);
}


/*
 * Get size info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a integer number.
 *
 */
static PyObject*
MKLNdarray_get_size(MKLNdarray *self, void *closure) {
    size_t total_element = 1;
    if (self->nd <= 0) {
        total_element = 0;
    } else {
        for (int i = 0; i < self->nd; i++) {
            total_element *= self->user_structure[i];
        }
    }
    return PyInt_FromLong(total_element);
}


/*
 * Get base info of a MKLNdarray instance.
 *
 * Register in MKLNdarray_getset
 *
 * Return a PyObject.
 *
 */
static PyObject*
MKLNdarray_get_base(MKLNdarray *self, void *closure) {
    PyObject * base = self->base;
    if (!base) {
        base = Py_None;
    }
    Py_INCREF(base);
    return base;
}



/*
 * Create a PyArrayObject from a MKLNdarray.
 */
PyObject* MKLNdarray_CreateArrayObj(MKLNdarray *self) {

    if (self->nd < 0 ||
        self->private_data == NULL ||
        self->private_layout == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Can't convert from a uninitialized MKLNdarray");
        return NULL;
    }

    npy_intp npydims[MAX_NDIM] = {0};

    for (int i = 0; i < self->nd; i++) {
        npydims[i] = (npy_intp)self->user_structure[i];
    }

    PyArrayObject* rval = NULL;
    if (self->dtype == MKL_FLOAT64) {
        // float64
        rval = (PyArrayObject*)PyArray_SimpleNew(self->nd, npydims, NPY_FLOAT64);
    } else {
        // float32
        rval = (PyArrayObject*)PyArray_SimpleNew(self->nd, npydims, NPY_FLOAT32);
    }

    if (!rval) {
        return NULL;
    }

    void* rval_data = PyArray_DATA(rval);
    dnnLayout_t layout_user = NULL;
    int status = -1;
    dnnPrimitive_t primitive = NULL;

    size_t mkl_size[MAX_NDIM] = {0};
    size_t mkl_stride[MAX_NDIM] = {0};

    // nchw -> whcn
    for (int i = 0; i < self->nd; i++) {
        mkl_size[i] = (MKLNdarray_DIMS(self))[self->nd - i - 1];
        mkl_stride[i] = (MKLNdarray_STRIDES(self))[self->nd - i -1];
    }

    if (self->dtype == MKL_FLOAT64) { // float64
        status = dnnLayoutCreate_F64(&layout_user,
                                     self->nd,
                                     mkl_size,
                                     mkl_stride);

        if (status != 0 || layout_user == NULL) {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnLayoutCreate_F64 failed");
            Py_DECREF(rval);
            return NULL;
        }

        status = dnnConversionCreate_F64(&primitive, self->private_layout, layout_user);
        if (status != 0 || primitive == NULL) {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnConversionCreate_F64 failed");
            Py_DECREF(rval);
            return NULL;
        }

        status = dnnConversionExecute_F64(primitive, (void*)self->private_data, (void*)rval_data);
        if (status != 0) {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnExecute_F64 failed");
            Py_DECREF(rval);
            return NULL;
        }
    } else {  // float32
        status = dnnLayoutCreate_F32(&layout_user,
                                     self->nd,
                                     mkl_size,
                                     mkl_stride);

        if (status != 0 || layout_user == NULL) {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnLayoutCreate_F32 failed");
            Py_DECREF(rval);
            return NULL;
        }

        status = dnnConversionCreate_F32(&primitive, self->private_layout, layout_user);
        if (status != 0 || primitive == NULL) {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnConversionCreate_F32 failed, %d", status);
            Py_DECREF(rval);
            return NULL;
        }

        status = dnnConversionExecute_F32(primitive, (void*)self->private_data, (void*)rval_data);
        if (status != 0) {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnExecute_F32 failed");
            Py_DECREF(rval);
            return NULL;
        }
    }

    return (PyObject*)rval;
}


/*
 * Create a new MKLNdarray instance and set all elements to zero.
 * This function will be called when do MKLNdarray.zeros(shape, typenum) in python code.
 *
 * shape: a tuple contains shape info. length of shape should <= MAX_NDIM
 * typenum: 11 means float32, 12 means float64. float32 by default.
 *
 * This function will call MKLNdarray_ZEROS to do detailed processing.
 *
 */
PyObject* MKLNdarray_Zeros(PyObject *_unused, PyObject *args) {
    if (!args) {
        PyErr_SetString(PyExc_TypeError, "MKLNdarray_Zeros: function takes at least 1 argument");
        return NULL;
    }

    PyObject* shape = NULL;
    int typenum = -1;

    // pas
    if (!PyArg_ParseTuple(args, "O|i", &shape, &typenum)) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_Zeros: PyArg_ParseTuple failed \n");
        return NULL;
    }

    if (typenum != MKL_FLOAT32 && typenum != MKL_FLOAT64) {
        printf("No dtype is specified. Use float32 as default. \n");
        typenum = 11;
    }

    if (!PySequence_Check(shape)) {
        PyErr_SetString(PyExc_TypeError, "shape argument must be a sequence");
        return NULL;
    }

    int shplen = PySequence_Length(shape);
    if (shplen <= 0 || shplen > MAX_NDIM) {
        PyErr_Format(PyExc_TypeError, "length of shape argument must be 1 ~ %d", MAX_NDIM);
        return NULL;
    }

    size_t newdims[MAX_NDIM] = {0};
    for (int i = shplen -1; i >= 0; i--) {
        PyObject* shp_el_obj = PySequence_GetItem(shape, i);
        if (shp_el_obj == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_Zeros: index out of bound in sequence");
            return NULL;
        }

        int shp_el = PyInt_AsLong(shp_el_obj);
        Py_DECREF(shp_el_obj);

        if (shp_el < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "MKLNdarray_Zeros: shape must contain only non-negative values for size of a dimension");
            return NULL;
        }
        newdims[i] = (size_t)shp_el;
    }

    PyObject* rval = MKLNdarray_ZEROS(shplen, newdims, typenum);
    return (PyObject*)rval;
}


PyObject*
MKLNdarray_debug_print(MKLNdarray *self) {
    if (! MKLNdarray_Check((PyObject*)self)) {
        printf("Input Object is not a MKLNdarray instance. \n");
        return NULL;
    }

    if (self->nd < 0 || self->dtype < 0) {
        printf("Input MKLNdarray is not initialized. \n");
        return NULL;
    }

    printf("ndim : %d \n", self->nd);
    printf("dtype: %s \n", MKL_TYPE[self->dtype]);
    printf("shape: (");
    for (int i = 0; i < self->nd; i ++) {
        printf("%ld, ", self->user_structure[i]);
    }
    printf(")\n");

    printf("stride: (");
    for (int i = 0; i < self->nd; i ++) {
        printf("%ld, ", self->user_structure[i + self->nd]);
    }
    printf(")\n");
    printf("layout: %p\n", (void*)&(self->private_layout));
    printf("data  : %p\n", self->private_data);

    Py_INCREF(Py_None);
    return Py_None;
}


/*
 * type:tp_methods
 * Describe methos of a type. ml_name/ml_meth/ml_flags/ml_doc.
 * ml_name: name of method
 * ml_meth: PyCFunction, point to the C implementation
 * ml_flags: indicate how the call should be constructed
 * ml_doc: docstring
 *
 */
static PyMethodDef MKLNdarray_methods[] = {

    {"__array__",
        (PyCFunction)MKLNdarray_CreateArrayObj, METH_VARARGS,
        "Copy from MKL to a numpy ndarray."},
    /*
    {"__copy__",
        (PyCFunction)MKLNdarray_View, METH_NOARGS,
        "Create a shallow copy of this object. Used by module copy"},
    */
    {"zeros",
        (PyCFunction)MKLNdarray_Zeros, METH_STATIC | METH_VARARGS,
        "Create a new MklNdarray with specified shape, filled with zeros."},

    {"__debug__",
        (PyCFunction)MKLNdarray_debug_print, METH_VARARGS,
        "Print debug info of a MKLNdarray."},
    /*
    {"copy",
        (PyCFunction)MKLNdarray_Copy, METH_NOARGS,
        "Create a copy of this object."},
    */

    {NULL, NULL, 0, NULL}  /* Sentinel */
};


/* type:tp_members
 * Describe attributes of a type. name/type/offset/flags/doc.
 */
static PyMemberDef MKLNdarray_members[] = {
    {NULL}      /* Sentinel */
};


/*
 * type:tp_getset
 * get/set attribute of instances of this type. name/getter/setter/doc/closure
 *
 */
static PyGetSetDef MKLNdarray_getset[] = {
    {"shape",
        (getter)MKLNdarray_get_shape,
        NULL,
        "shape of this ndarray (tuple)",
        NULL},

    {"dtype",
        (getter)MKLNdarray_get_dtype,
        NULL,
        "the dtype of the element.",
        NULL},

    {"size",
        (getter)MKLNdarray_get_size,
        NULL,
        "the number of elements in this object.",
        NULL},

    {"ndim",
        (getter)MKLNdarray_get_ndim,
        NULL,
        "the number of dimensions in this objec.",
        NULL},

    {"base",
        (getter)MKLNdarray_get_base,
        NULL,
        "if this ndarray is a view, base is the original ndarray.",
        NULL},

    {NULL, NULL, NULL, NULL}  /* Sentinel*/
};

/*
 * type object.
 * If you want to define a new object type, you need to create a new type object.
 *
 */
static PyTypeObject MKLNdarrayType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "MKLNdarray",              /*tp_name*/
    sizeof(MKLNdarray),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MKLNdarray_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    MKLNdarray_repr,           /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
#if PY_MAJOR_VERSION >= 3
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, /*tp_flags*/
#endif
    "MKLNdarray objects",      /*tp_doc */
    0,                         /*tp_traverse*/
    0,                         /*tp_clear*/
    0,                         /*tp_richcompare*/
    0,                         /*tp_weaklistoffset*/
    0,                         /*tp_iter*/
    0,                         /*tp_iternext*/
    MKLNdarray_methods,        /*tp_methods*/
    MKLNdarray_members,        /*tp_members*/
    MKLNdarray_getset,         /*tp_getset*/
    0,                         /*tp_base*/
    0,                         /*tp_dict*/
    0,                         /*tp_descr_get*/
    0,                         /*tp_descr_set*/
    0,                         /*tp_dictoffset*/
    (initproc)MKLNdarray_init, /*tp_init*/
    0,                         /*tp_alloc*/
    MKLNdarray_new,            /*tp_new*/
};



/*
 * Check an input is an instance of MKLNdarray or not.
 * Same as PyArray_Check
 *
 */
int MKLNdarray_Check(const PyObject *ob) {
    return ((Py_TYPE(ob) == &MKLNdarrayType) ? 1 : 0);
}


/*
 * Try to new a MKLNdarray instance.
 * This function is different from MKLNdarray_new which is set for tp_new.
 *
 * MKLNdarray_new is call by python when python has allocate memory for it already.
 * MKLNdarray_New will be called manually in C/C++ code, so we need to call tp_alloc manually.
 *
 */
PyObject*
MKLNdarray_New(int nd, int typenum) {

    if (nd < 0 || nd > MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                        "MKLNdarray does not support a %d-dim array. Try array which ndim is <= %d", nd, MAX_NDIM);
        return NULL;
    }

    MKLNdarray* self = (MKLNdarray*)(MKLNdarrayType.tp_alloc(&MKLNdarrayType, 0));
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_New failed to allocate self");
        return NULL;
    }

    self->base              = NULL;
    self->nd                = nd;
    self->dtype             = typenum;
    self->private_data      = NULL;
    self->private_workspace = NULL;
    self->data_size         = 0;
    self->private_layout    = NULL;
    memset((void*)(self->user_structure), 0, 2 * MAX_NDIM * sizeof (size_t));

    return (PyObject*)self;
}



/*
 * Declare methods belong to this module but not in MKLNdarray.
 * Used in module initialization function.
 *
 * Users can access these methods by module.method_name() after they have imported this module.
 *
 */
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}   /* Sentinel */
};


/*
 * Module initialization function.
 * TODO: Just for Python2.X. Need to add support for Python3.
 */
PyMODINIT_FUNC
initmkl_ndarray(void) {
    import_array();
    PyObject* m = NULL;

    if (PyType_Ready(&MKLNdarrayType) < 0) {
        printf("MKLNdarrayType failed \n");
        return;
    }

    // add attribute to MKLNdarrayType
    // if user has import MKLNdarrayType already, they can get typenum of float32 and float64
    // by MKLNdarray.float32 or MKLNdarray.float64
    PyDict_SetItemString(MKLNdarrayType.tp_dict, "float32", PyInt_FromLong(MKL_FLOAT32));
    PyDict_SetItemString(MKLNdarrayType.tp_dict, "float64", PyInt_FromLong(MKL_FLOAT64));

    m = Py_InitModule3("mkl_ndarray", module_methods, "MKL implementation of a ndarray object.");
    if (m == NULL) {
        printf("Py_InitModule3 failed to init mkl_ndarray. \n");
        return;
    }
    Py_INCREF(&MKLNdarrayType);
    PyModule_AddObject(m, "MKLNdarray", (PyObject*)&MKLNdarrayType);

    return;
}

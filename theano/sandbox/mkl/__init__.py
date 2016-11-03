from __future__ import absolute_import, print_function, division
import logging
import os
import shutil
import stat
import sys

import theano
from theano import config, gof 

_logger_name = 'theano.sandbox.mkl'
_logger = logging.getLogger(_logger_name)

# This variable is True by default, and set to False if mkl library
# is not found on the platform
mkl_available = True

class getMKLBuildDate(gof.op):
    def c_headers(self):
        return ['mkl.h']

    def c_libraries(self):
        return ['mkl_rt']

    def make_node(self):
        return gof.Apply(self, [], [gof.Generic()()])

    def c_code(self, node, name, inputs, outputs, sub):
        o = outputs[0]
        return textwrap.dedent(
            """
            MKLVersion v;
            mkl_get_version(&v);
            %(o)s = atoi(v.Build);
            """) % locals()

    def c_code_cache_version(self):
        # Not needed, but make it clear that we do not want to cache this.
        return None


def mkl_build_date():
    """
    Return the current mkl build date (e.g., 20160701) we compile with.
    """
    if not mkl_available():
        raise Exception(
            "We can't determine the mkl build date as it is not available",
            mkl_available.msg)

    if mkl_build_date.v is None:
        f = theano.function([], getMKLBuildDate()(),
                            theano.Mode(optimizer=None),
                            profile=False)
        mkl_build_date.v = f()
    return mkl_build_date.v
mkl_build_date.v = None


def mkl_available():
    if config.dnn.enabled == "cuDNN":
        mkl_available.avail = False
        mkl_available.msg = "Disabled by dnn.enabled flag"
        return mkl_available.avail
    
    if config.dnn.enabled == "auto":
        if config.device != "cpu":
            mkl_available.avail = False
            mkl_available.msg = "Mkl is disabled since device is not CPU"
            return mkl_available.avail
        else:
            ## FIXME, need ICC_Compiler support
            preambule = textwrap.dedent(
                """
                #include <stdio.h>
                #include <mkl.h>
                """)
    
            body = textwrap.dedent(
                """
                cudnnHandle_t _handle = NULL;
                cudnnStatus_t err;
                if ((err = cudnnCreate(&_handle)) != CUDNN_STATUS_SUCCESS) {
                  fprintf(stderr, "could not create cuDNN handle: %s",
                          cudnnGetErrorString(err));
                  return 1;
                }
                """)
            # to support path that includes spaces, we need to wrap it with double quotes on Windows
            path_wrapper = "\"" if os.name =='nt' else ""
            params = ["-l", "cudnn"]
            params.extend(['-I%s%s%s' % (path_wrapper, os.path.dirname(__file__), path_wrapper)])
            if config.dnn.include_path:
                params.extend(['-I%s%s%s' % (path_wrapper, config.dnn.include_path, path_wrapper)])
            if config.dnn.library_path:
                params.extend(['-L%s%s%s' % (path_wrapper, config.dnn.library_path, path_wrapper)])
            if config.nvcc.compiler_bindir:
                params.extend(['--compiler-bindir',
                               '%s%s%s' % (path_wrapper, config.nvcc.compiler_bindir, path_wrapper)])
            params.extend([flag for flag in config.nvcc.flags.split(' ') if flag])
    
            # Do not run here the test program. It would run on the
            # default gpu, not the one selected by the user. If mixed
            # GPU are installed or if the GPUs are configured in
            # exclusive mode, this cause bad detection.
            comp, out, err = nvcc_compiler.NVCC_compiler.try_flags(
                flag_list=params, preambule=preambule, body=body,
                try_run=False, output=True)
    
            mkl_available.avail = comp
            if not mkl_available.avail:
                mkl_available.msg = (
                    "Can not compile with cuDNN. We got this error:\n" +
                    str(err))
            else:
                # If we can compile, check that we can import and run.
                v = dnn_version()
                if isinstance(v, tuple) and v[0] != v[1]:
                    mkl_available.avail = False
                    mkl_available.msg = ("Mixed dnn version. The header is"
                                         " from one version, but we link with"
                                         " a different version %s" % str(v))
                    raise RuntimeError(mkl_available.msg)
                if v == -1 or v[0] < 4007:
                    # 4007 is the final release of cudnn v4
                    mkl_available.avail = False
                    mkl_available.msg = "Version is too old. Update to v5, was %d." % v[0]
                    raise RuntimeError(mkl_available.msg)
                else:
                    mkl_available.avail = comp

    if config.dnn.enabled == "mkl":
        if not mkl_available.avail:
            raise NotImplemented(
                "mkl is not supported, %s" % mkl_available.msg)

    ## leave mkl-dnn here for future use
    if config.dnn.enabled == "mkl-dnn":
        if not mkl_available.avail:
            raise NotImplemented(
                "mkl-dnn is not supported, %s" % mkl_available.msg)
    return mkl_available.avail

mkl_available.avail = None
mkl_available.msg = None

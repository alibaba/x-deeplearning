# coding: utf-8
from __future__ import absolute_import
import ctypes
import numpy as np

from .base import check_call, LIB, c_str, c_array, blaze2numpy, numpy2blaze, numpy_dtype_size

class Predictor(object):
    """ The blaze predictor 
    """
    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        check_call(LIB.Blaze_PredictorDelete(self.handle))

    def reshape_input(self, name, shape):
        check_call(LIB.Blaze_PredictorReshapeInput(self.handle,
                                                  c_str(name),
                                                  c_array(ctypes.c_int, shape),
                                                  len(shape)))

    def feed(self, name, data):
        data_len = ctypes.c_size_t(numpy_dtype_size(data) * data.size)
        check_call(LIB.Blaze_PredictorFeed(self.handle,
                                           c_str(name),
                                           data.ctypes.data_as(ctypes.c_void_p),
                                           data_len))

    def forward(self):
        check_call(LIB.Blaze_PredictorForward(self.handle))

    def output_shape(self, name):
        ndim = ctypes.c_size_t()
        pdata = ctypes.POINTER(ctypes.c_size_t)()
        check_call(LIB.Blaze_PredictorOutputShape(self.handle,
                                                  c_str(name),
                                                  ctypes.byref(ndim),
                                                  ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])

    def list_input_names(self):
        ndim = ctypes.c_size_t()
        pdata = ctypes.POINTER(ctypes.c_char_p)()
        check_call(LIB.Blaze_PredictorListInputNames(self.handle,
                                                     ctypes.byref(ndim),
                                                     ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])
    
    def list_output_names(self):
        ndim = ctypes.c_size_t()
        pdata = ctypes.POINTER(ctypes.c_char_p)()
        check_call(LIB.Blaze_PredictorListOutputNames(self.handle,
                                                      ctypes.byref(ndim),
                                                      ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])

    def output_dtype(self, name):
        dtype = ctypes.c_int()
        check_call(LIB.Blaze_PredictorOutputDataType(self.handle,
                                                     c_str(name),
                                                     ctypes.byref(dtype)))
        return blaze2numpy[dtype.value]

    def output_asnumpy(self, name):
        data = np.empty(self.output_shape(name), dtype=self.output_dtype(name))
        check_call(LIB.Blaze_PredictorOutput(self.handle,
                                             c_str(name),
                                             data.ctypes.data_as(ctypes.c_void_p),
                                             ctypes.c_size_t(data.size)))
        return data

    def list_internal_names(self):
        ndim = ctypes.c_size_t()
        pdata = ctypes.POINTER(ctypes.c_char_p)()
        check_call(LIB.Blaze_PredictorParamName(self.handle,
                                                ctypes.byref(ndim),
                                                ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])

    def internal_shape(self, name):
        ndim = ctypes.c_size_t()
        pdata = ctypes.POINTER(ctypes.c_size_t)()
        check_call(LIB.Blaze_PredictorParamShape(self.handle,
                                                 c_str(name),
                                                 ctypes.byref(ndim),
                                                 ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])

    def internal_dtype(self, name):
        dtype = ctypes.c_int()
        check_call(LIB.Blaze_PredictorParamDataType(self.handle,
                                                    c_str(name),
                                                    ctypes.byref(dtype)))
        return blaze2numpy[dtype.value]

    def internal_asnumpy(self, name):
        data = np.empty(self.internal_shape(name), dtype=self.internal_dtype(name))
        check_call(LIB.Blaze_PredictorParam(self.handle,
                                            c_str(name),
                                            data.ctypes.data_as(ctypes.c_void_p),
                                            ctypes.c_size_t(data.size)))
        return data

    def register_observers(self, observer_names):
        """ current available names: cost/profile
        """
        names = []
        for name in observer_names:
            names.append(c_str(name))
        check_call(LIB.Blaze_PredictorRegisterObservers(self.handle,
                                                        len(names),
                                                        c_array(ctypes.c_char_p, names)))

    def dump_observers(self):
        ndim = ctypes.c_size_t()
        pkey = ctypes.POINTER(ctypes.c_char_p)()
        pvalue = ctypes.POINTER(ctypes.c_char_p)()
        check_call(LIB.Blaze_PredictorDumpObservers(self.handle,
                                                    ctypes.byref(ndim),
                                                    ctypes.byref(pkey),
                                                    ctypes.byref(pvalue)))
        omap = { }
        for x in xrange(ndim.value):
            omap[pkey[x]] = pvalue[x]
        return omap

class PredictorManager(object):
    """The predictor manager.
    """
    def __init__(self):
        self.handle = ctypes.c_void_p()
        check_call(LIB.Blaze_PredictorManagerCreate(ctypes.byref(self.handle)))

    def __del__(self):
        check_call(LIB.Blaze_PredictorManagerDelete(self.handle))

    def set_data_type(self, data_type=np.float32):
        check_call(LIB.Blaze_PredictorManagerSetDataType(self.handle, numpy2blaze[data_type]))

    def set_run_mode(self, run_mode):
        check_call(LIB.Blaze_PredictorManagerSetRunMode(self.handle, c_str(run_mode)))

    def load_sparse_model_weight(self, blaze_sparse_model_weight_file):
        check_call(LIB.Blaze_LoadSparseModelWeight(self.handle, c_str(blaze_sparse_model_weight_file)))

    def load_model(self, blaze_model_file, optimization_pass):
        check_call(LIB.Blaze_PredcitorManagerLoadModel(self.handle, c_str(blaze_model_file), optimization_pass))

    def load_deepnet_model(self, conf_file, param_file, optimization_pass):
        check_call(LIB.Blaze_PredictorManagerLoadDeepNetModel(self.handle, c_str(conf_file),
                                                             c_str(param_file), optimization_pass))

    def create_predictor(self, device_type, device_id):
        """ 0 -> CPU   1 -> GPU
        """
        predictor_handle = ctypes.c_void_p()
        check_call(LIB.Blaze_PredictorCreate(self.handle, device_type, device_id,
                                            ctypes.byref(predictor_handle)))
        return Predictor(predictor_handle)

def init_scheduler(enable_batching,
                   max_batch_size,
                   batch_timeout_micros,
                   num_threads_for_cpu,
                   num_threads_for_cuda,
                   num_threads_for_pipe):
  check_call(LIB.Blaze_InitScheduler(1 if enable_batching else 0,
                                     max_batch_size,
                                     batch_timeout_micros,
                                     num_threads_for_cpu,
                                     num_threads_for_cuda,
                                     num_threads_for_pipe))


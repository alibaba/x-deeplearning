# coding: utf-8
from __future__ import absolute_import
import ctypes

from .base import LIB, check_call, c_str, c_array

def Convert2Blaze(model_conf_path,
                  model_data_path,
                  model_type,
                  weight_type,
                  io_type,
                  x_type,
                  blaze_path,
                  tobinary=1):
    """ convert onnx/tensorflow/mxnet/xdl/ulf model to blaze internal.

    Parameter:
      model_conf_path:
         The model conf path, such as mxnet's json graph
      
      model_data_path:
         The model data path, such as mxnet's ndarray map
      
      model_type:
         1 --> ulf  2 --> onnx 3 --> mxnet 4 ---> tensorflow  5 ---> xdl  6 ---> xdl_ulf 
      
      weight_type:
         The model weight type, 1 -> kFloat  12 -> kFloat16
      
      io_type:
         The model data type, 1 -> kFloat, 12 ->kFloat16
      
      x_type:
         The x data_type for many op types. op_type -> data_type map
      
      blaze_path:
         The path for saving blaze model
      
      tobinary:
         Whether to save as binary, 1 save as binary, 0 save as text
    """
    x_type_key = []
    x_type_value = []
    for key, value in x_type.iteritems():
      x_type_key.append(c_str(key))
      x_type_value.append(value)

    check_call(LIB.Blaze_ConvertBlaze(c_str(model_conf_path),
                                      c_str(model_data_path),
                                      model_type,
                                      weight_type,
                                      io_type,
                                      len(x_type_key),
                                      c_array(ctypes.c_char_p, x_type_key),
                                      c_array(ctypes.c_int, x_type_value),
                                      c_str(blaze_path),
                                      tobinary))

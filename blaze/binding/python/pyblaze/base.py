""" base for blaze """
from __future__ import absolute_import

import sys
import os
import ctypes
import numpy as np
from enum import Enum
from .libinfo import find_lib_path

#------------------------------------------
# library loading
#------------------------------------------
py_str = lambda x: x

class BlazeError(Exception):
    """Error that will be throwed by all The Blaze functions"""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def _load_lib(name):
    """ Load library by searching possible path. """
    libpath = find_lib_path(name)
    lib = ctypes.CDLL(libpath, ctypes.RTLD_GLOBAL)
    return lib

# library instance of xdl
LIB = _load_lib('blaze')

blaze2numpy = {
  1 : np.float32,
  12 : np.float16,
  2 : np.int32,
  10 : np.int64,
  5 : np.bool_
}
numpy2blaze = {
  np.float32 : 1,
  np.float16 : 12,
  np.int32 : 2,
  np.int64 : 10,
  np.bool_ : 5
}

def numpy_dtype_size(data):
  if data.dtype == np.float32:
    return ctypes.sizeof(ctypes.c_float)
  elif data.dtype == np.float16:
    return ctypes.sizeof(ctypes.c_int16)
  elif data.dtype == np.int64:
    return ctypes.sizeof(ctypes.c_int64)
  elif data.dtype == np.int32:
    return ctypes.sizeof(ctypes.c_int32)
  elif data.dtype == np.bool_:
    return ctypes.sizeof(ctypes.c_bool)
  else:
    return ctypes.sizeof(ctypes.c_float)

#-------------------------------------------
# helper function definition
#-------------------------------------------
def check_call(ret):
    """
    Check the return value of C API call

    This function will raise exception when error occurs
    Warp every API call with this function

    Parameters
    ---------
    ret : int
        return value from C API calls
    """
    if ret != 0:
        msg = ctypes.c_char_p(None)
        LIB.Blaze_GetLastErrorString(ctypes.byref(msg))
        raise BlazeError(py_str(msg.value))

def c_str(string):
    """
    Create ctypes char* from a python string

    Parameters:
    ---------
    string : string type
             python string

    Returns:
    ---------
    str: c_char_p
       A char pointer that can be passed to C API
    """
    return ctypes.c_char_p(string.encode('utf-8'))

def c_array(ctype, values=None, count=0):
    """
    Create ctypes array from a python array

    Parameters:
    ----------
    ctype: ctypes data type
        data type of the array we want to convert to
    values: tuple or list
        data content

    Returns:
    ---------
      out : ctypes array
         Created ctypes array
    """
    if values is not None:
        return (ctype * len(values))(*values)
    elif count > 0:
        return (ctype * count)()



# coding: utf-8
from __future__ import absolute_import
import ctypes

from .base import LIB, check_call, c_str, c_array

def Optimize(raw_model_file,
             optimized_model_file,
             tobinary = 1):
    """ Optimize raw blaze graph to optimized graph.

    Parameter:
      raw_model_file:
         The raw model file in blaze format
      
      optimized_model_file:
         The optimized model file in blaze format
      
      tobinary:
         Whether the optimized model is binary format.
    """
    check_call(LIB.Blaze_OptimizeBlaze(c_str(raw_model_file),
                                       c_str(optimized_model_file),
                                       tobinary))

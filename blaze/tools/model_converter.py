# coding: utf-8
from __future__ import absolute_import

from pyblaze.base import BlazeError
from pyblaze.converter import Convert2Blaze
import numpy as np
import logging
import sys, getopt

def convert_model(conf_file,
                  data_file,
                  model_type,
                  weight_type,
                  io_type,
                  x_type,
                  out_file,
                  is_binary=1):
  """ A model converter for mxnet/ulf/tensorflow/onnx/xdl -> blaze internal.

   Parameter
   ---------
      conf_file:
        The model conf 
      
      data_file:
        The model data
      
      model_type:
        1 --> ulf  2 --> onnx 3 --> mxnet 4 ---> tensorflow  5 ---> xdl  6 ---> xdl_ulf 
      
      weight_type:
        The model weight type, 1 -> kFloat  12 -> kFloat16
      
      io_type:
        The model data type, 1 -> kFloat, 12 ->kFloat16
      
      x_type:
        The x data_type for many op types.
      
      out_file:
        the blaze internal file

   Return
   ------
      True if model convert success, else False.
  """
  try:
    print 'conf_file=', conf_file, ' data_file=',  data_file, ' model_type=', model_type, ' weight_type=', weight_type, ' io_type=', io_type, ' x_type=', x_type, ' out_file=', out_file, ' is_binary=', is_binary

    Convert2Blaze(conf_file,
                  data_file,
                  model_type,
                  weight_type,
                  io_type,
                  x_type,
                  out_file,
                  is_binary)
  except BlazeError as e:
    logging.error("convert model failed,%s" % (e.value))
    return False
  else:
    return True

def print_help():
    print 'model_converter.py -c <model_conf_file> -d <model_data_file> -o <output_file> -t <model_type> -w <weight_type> -i <io_type> -x <x_type> -b <binary>'
    print ''
    print '       model_type option:'
    print '            1 -> ULF  2 -> ONNX 3 -> MXNet 4 -> TensorFlow  5 -> XDL 6 -> XDL-ULF'
    print ''
    print '       data_type option:'
    print '            1 -> float32  12 -> float16'
    print ''

if __name__ == '__main__':
  try:
    opts, args = getopt.getopt(sys.argv[1:],"hc:d:o:t:w:i:x:b:",
                               ["cfile=", "dfile=", "ofile=", "model_type=", "weight_type=", "io_type=", "x_type=", "binary="])
  except getopt.GetoptError:
    print_help()
    sys.exit(2)

  conf_file = ""
  data_file = ""
  model_type = 6  ## xdl_ulf
  weight_type = 1 ## kFloat
  io_type = 1 ## kFloat
  x_type = { }
  out_file = ""
  is_binary = 1

  for opt, arg in opts:
    if opt == '-h':
      print_help()
      sys.exit()
    elif opt in ("-c", "--cfile"):
      conf_file = arg
    elif opt in ("-d", "--dfile"):
      data_file = arg
    elif opt in ("-o", "--ofile"):
      out_file = arg
    elif opt in ("-t", "--model_type"):
      model_type = int(arg)
    elif opt in ("-w", "--weight_type"):
      weight_type = int(arg)
    elif opt in ("-w", "--io_type"):
      io_type = int(arg)
    elif opt in ("-x", "--x_type"):
      x_type = { }
      segments = split(arg, ',')
      for item in segments:
        splits = split(item, ":")
        x_type[splits[0]] = int(splits[1])
    elif opt in ("-b", "--binary"):
      is_binary = int(arg)

  if convert_model(conf_file = conf_file,
                   data_file = data_file,
                   model_type = model_type,
                   weight_type = weight_type,
                   io_type = io_type,
                   x_type = x_type,
                   out_file = out_file,
                   is_binary = is_binary):
    print "convert success"
    sys.exit(0)
  else:
    print "convert model:", conf_file, " ", data_file, " failed"
    sys.exit(1)


# coding: utf-8
from __future__ import absolute_import

from pyblaze.base import BlazeError
from pyblaze.optimizer import Optimize
import numpy as np
import logging
import sys, getopt

def print_help():
  print 'model_optimizer -i <infile> -o <ofile> -b <binary>'
  print ''

if __name__ == '__main__':
  try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:b:",
                               ["ifile=", "ofile=", "binary="])
  except getopt.GetoptError:
    print_help()
    sys.exit(2)
  
  i_file = ''
  o_file = ''
  to_binary = 1

  for opt, arg in opts:
    if opt == '-h':
      print_help()
      sys.exit()
    elif opt in ("-i", "--ifile"):
      i_file = arg
    elif opt in ("-o", "--ofile"):
      o_file = arg
    elif opt in ("-b", "--binary"):
      to_binary = int(arg)

  try:
    Optimize(i_file, o_file, to_binary)
  except BlazeError as e:
    logging.error("optimized model failed, msg=%s" % (e.value))
    sys.exit(1)


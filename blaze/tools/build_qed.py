# coding: utf-8
from __future__ import absolute_import

from pyblaze.base import BlazeError
from pyblaze.qed import sparse2qed_fp32, sparse2qed_fp16
import numpy as np
import logging
import sys, getopt

def print_help():
  print 'build_qed -p <path> -i <infile> -o <ofile> -s <precision> -t <threadnum>'
  print ''
  print '      precision option: 0 -> fp32 1 -> fp16'

if __name__ == '__main__':
  try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:t:s:p:",
                               ["ifile=", "ofile=", "threadnum=", "precision=", "path="])
  except getopt.GetoptError:
    print_help()
    sys.exit(2)
  
  path = ''
  i_file = ''
  o_file = ''
  precision = 1  # fp16
  thread_num = 1

  for opt, arg in opts:
    if opt == '-h':
      print_help()
      sys.exit()
    elif opt in ("-p", "--path"):
      path = arg
    elif opt in ("-i", "--ifile"):
      i_file = arg
    elif opt in ("-o", "--ofile"):
      o_file = arg
    elif opt in ("-s", "--precision"):
      precision = int(arg)
    elif opt in ("-t", "--threadnum"):
      thread_num = int(arg)

  if precision == 1:
    sparse2qed_fp16(path, i_file, o_file, thread_num)
  elif precision == 0:
    sparse2qed_fp32(path, i_file, o_file, thread_num)
  else:
    print 'unkown presicion:', precision
    sys.exit(1)

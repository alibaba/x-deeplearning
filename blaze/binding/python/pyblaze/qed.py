# coding: utf-8
from __future__ import absolute_import
import ctypes

from .base import LIB, check_call, c_str

def sparse2qed_fp32(path, meta, output_file, thread_num):
    """ convert sparse raw param -> fp32 qed bin
    """
    check_call(LIB.Blaze_QuickEmbeddingBuildFp32(c_str(path),
                                                 c_str(meta),
                                                 c_str(output_file),
                                                 thread_num))

def sparse2qed_fp16(path, meta, output_file, thread_num):
    """ convert sparse raw param -> fp16 qed bin
    """
    check_call(LIB.Blaze_QuickEmbeddingBuildFp16(c_str(path),
                                                 c_str(meta),
                                                 c_str(output_file),
                                                 thread_num))

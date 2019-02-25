from __future__ import absolute_import

import os
import platform

def find_lib_path(name):
    """ Find PS related dynamic library files for setup

    Returns:
    ------
    lib_path: list(string)

    list of all found path to the libraries
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    
    product_path = os.path.abspath(os.path.join(curr_path, './'))
    lib_path = os.path.join(product_path, 'lib%s.so'%name)
    if os.path.exists(lib_path):
        return lib_path 

    product_path = os.path.abspath(os.path.join(curr_path, '../../lib64/'))
    lib_path = os.path.join(product_path, 'lib%s.so'%name)
    if os.path.exists(lib_path):
        return lib_path 

    build_path = os.path.abspath(os.path.join(curr_path, '../../../build/blaze/'))
    lib_path = os.path.join(build_path, 'lib%s.so'%name)
    if os.path.exists(lib_path):
        return lib_path 
    
    AssertionError('can not found lib %s' % name)

# current version
__version__ = "0.1"

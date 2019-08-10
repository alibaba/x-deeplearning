/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "pybind11/pybind11.h"
#include "xdl/python/pybind/op_define_wrapper.h"
#include "xdl/python/pybind/core_wrapper.h"
#include "xdl/python/pybind/executor_wrapper.h"
#include "xdl/python/pybind/pyfunc_op.h"
#include "xdl/python/pybind/data_io_wrapper.h"
#include "xdl/python/pybind/model_server_wrapper.h"

using namespace xdl::python_lib;

PYBIND11_MODULE(libxdl_python_pybind, m) {
  m.doc() = "xdl python wrapper";

  CorePybind(m);

  OpDefinePybind(m);

  ExecutorPybind(m);

  PyFuncPybind(m);

  DataIOPybind(m);

#ifdef USE_PS_PLUS
  ModelServerPybind(m);
#endif
}


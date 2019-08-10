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

#include "xdl/python/pybind/model_server_wrapper.h"

#include "xdl/core/utils/logging.h"

#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/ops/ps_ops/model_server/model_server.h"

namespace xdl {
namespace python_lib {

void ModelServerPybind(pybind11::module& m) {
  pybind11::class_<ModelServer>(m, "ModelServer")
    .def(pybind11::init<std::string, int, int, std::string, std::string>())
    .def("init", &ModelServer::Init)
    .def("forward_handle", &ModelServer::ForwardHandle)
    .def("backward_handle", &ModelServer::BackwardHandle);
}

}  // namespace python_lib
}  // namespace xdl


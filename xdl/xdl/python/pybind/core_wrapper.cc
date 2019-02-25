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

#include "xdl/python/pybind/core_wrapper.h"

#include <glog/logging.h>

#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "pybind11/numpy.h"

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/backend/device_singleton.h"

#define ONE_ARG(...) __VA_ARGS__
PYBIND11_MAKE_OPAQUE(ONE_ARG(std::unordered_map<std::string, std::string>));
PYBIND11_MAKE_OPAQUE(std::vector<xdl::Tensor>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<size_t>);

namespace xdl {
namespace python_lib {

void log_to_stderr() {
  FLAGS_logtostderr = 1;
}

void CorePybind(pybind11::module& m) {
  pybind11::class_<Status> status(m, "Status");
  status
    .def(pybind11::init<>())
    .def(pybind11::init<Status::ErrorCode, const std::string&>())
    .def_property_readonly("code", &Status::Code)
    .def_property_readonly("msg", &Status::Msg);

  pybind11::enum_<Status::ErrorCode>(status, "ErrorCode")
    .value("OK", Status::ErrorCode::kOk)
    .value("ArgumentError", Status::ErrorCode::kArgumentError)
    .value("IndexOverflow", Status::ErrorCode::kIndexOverflow)
    .value("PsError", Status::ErrorCode::kPsError)
    .value("OutOfRange", Status::ErrorCode::kOutOfRange)
    .value("Internal", Status::ErrorCode::kInternal);

  pybind11::class_<Tensor>(m, "Tensor", pybind11::buffer_protocol())
    .def("__init__", [](Tensor& m, pybind11::array arr) {
      DataType type = DataType::kFloat;
      int itemsize = arr.dtype().itemsize();
      switch(arr.dtype().kind()) {
      case 'f':
        if (itemsize == 4) {
          type = DataType::kFloat;
        } else {
          type = DataType::kDouble;
        }
        break;
      case 'i':
        if (itemsize == 1) {
          type = DataType::kInt8;
        } else if (itemsize == 4) {
          type = DataType::kInt32;
        } else {
          type = DataType::kInt64;
        }
        break;        
      default:
        XDL_LOG(FATAL) << "unsupported numpy type:" << arr.dtype().kind();
      }
      
      std::vector<size_t> dims;
      for (size_t i = 0; i < arr.ndim(); ++i) {
        dims.push_back(arr.shape(i));
      }

      Tensor t(DeviceSingleton::CpuInstance(), TensorShape(dims), type);
      memcpy(t.Raw<void>(), arr.data(), arr.nbytes());
      new (&m) Tensor(t);
    })
    .def_property_readonly("initialized", &Tensor::Initialized)
    .def_property_readonly("shape", &Tensor::Shape)
    .def_property_readonly("type", &Tensor::Type)
    .def_buffer([](Tensor& tensor) -> pybind11::buffer_info {
      XDL_TYPE_CASES(tensor.Type(), {
        std::vector<size_t> shape = tensor.Shape().Dims();
        std::vector<size_t> stride(shape.size());
        size_t k = sizeof(T);
        for (int i = shape.size() - 1; i >= 0; i--) {
          stride[i] = k;
          k *= shape[i];
        }
        return pybind11::buffer_info(
          tensor.Raw<T>(), sizeof(T),
          pybind11::format_descriptor<T>::format(),
          shape.size(), shape, stride);
      });
      CHECK(false) << "Unreachable code for Tensor numpy bind";
    });

  pybind11::bind_vector<std::vector<Tensor>>(
      m, "TensorVector");

  pybind11::bind_vector<std::vector<std::string>>(
      m, "StringVector");

  pybind11::bind_vector<std::vector<size_t>>(
      m, "SizeTVector");

  pybind11::bind_map<std::unordered_map<std::string, std::string>>(
      m, "StringStringMap");

  m.def("log_to_stderr", &log_to_stderr, "log to stderr");
}

}  // namespace python_lib
}  // namespace xdl




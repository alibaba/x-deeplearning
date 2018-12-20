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

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include <vector>

namespace xdl {
namespace python_lib {

struct PyFuncResult {
  struct TensorBuffer {
    std::string buf;
    std::vector<size_t> shape;
    DataType type;
  };
  Status status;
  std::vector<TensorBuffer> result;
};

}  // namespace xdl
}  // namespace python_lib

PYBIND11_MAKE_OPAQUE(std::vector<size_t>);
PYBIND11_MAKE_OPAQUE(std::vector<xdl::python_lib::PyFuncResult::TensorBuffer>);

namespace xdl {
namespace python_lib {

class PyObjectManager : public Singleton<PyObjectManager> {
 public:
  PyObjectManager() : counter_(0) {
    objects_ = new std::vector<pybind11::object>();
  }
  int64_t Set(pybind11::object obj) {
    std::unique_lock<std::mutex> lock(mu_);
    objects_->push_back(obj);
    return objects_->size() - 1;
  }
  Status Get(int64_t handle, pybind11::object* obj) {
    std::unique_lock<std::mutex> lock(mu_);
    if ((size_t)handle >= objects_->size()) {
      return Status::ArgumentError("Object Handle is invalid");
    }
    *obj = (*objects_)[handle];
	return Status::Ok();
  }
  std::mutex& Mutex() { return mu_; }
 private:
  std::mutex mu_;
  std::vector<pybind11::object>* objects_;
  int64_t counter_;
};

class _PyFuncOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    int64_t handle;
    XDL_CHECK_STATUS(ctx->GetAttr("handle", &handle));
    XDL_CHECK_STATUS(PyObjectManager::Instance()->Get(handle, &obj_));
    XDL_CHECK_STATUS(ctx->GetAttr("input_type", &input_type_));
    XDL_CHECK_STATUS(ctx->GetAttr("output_type", &output_type_));
    return Status::Ok();
  }
  Status Compute(OpKernelContext* ctx) override {
    std::vector<Tensor> inputs;
    XDL_CHECK_STATUS(ctx->GetInputList("input", &inputs));
    std::unique_lock<std::mutex> lock(PyObjectManager::Instance()->Mutex());
    pybind11::object obj_result = obj_(inputs);
    PyFuncResult* result = pybind11::cast<PyFuncResult*>(obj_result);
    XDL_CHECK_STATUS(result->status);
    std::vector<Tensor> outputs;
    XDL_CHECK_COND(result->result.size() == output_type_.size(),
        Status::ArgumentError("PyFunc Run Error, return size mismatch"));
    for (size_t i = 0; i < result->result.size(); i++) {
      XDL_CHECK_COND(result->result[i].type == output_type_[i],
        Status::ArgumentError("PyFunc Run Error, return type mismatch"));
    }
    for (auto&& buffer : result->result) {
      Tensor rst;
      XDL_CHECK_STATUS(ctx->Allocate(
            TensorShape(buffer.shape), buffer.type, &rst));
      Buffer* rst_buffer = rst.GetBuffer();
      memcpy(rst_buffer->begin(), &buffer.buf[0], rst_buffer->size());
      outputs.push_back(rst);
    }
    XDL_CHECK_STATUS(ctx->SetOutputList("output", outputs));
    return Status::Ok();
  }
 private:
  pybind11::object obj_;
  std::vector<DataType> input_type_;
  std::vector<DataType> output_type_;
};

XDL_DEFINE_OP(_PyFunc)
  .Attr("handle", AttrValue::kInt)
  .Attr("input_type", AttrValue::kDataTypeList)
  .Attr("output_type", AttrValue::kDataTypeList)
  .InputListV2("input", "input_type")
  .OutputListV2("output", "output_type");

XDL_REGISTER_KERNEL(_PyFunc, _PyFuncOp)
  .Device("CPU");

void PyFuncPybind(pybind11::module& m) {
  m.def("object_handle",
      [](pybind11::object obj)->int64_t{
        return PyObjectManager::Instance()->Set(obj);
      });

  pybind11::class_<PyFuncResult> py_func_result(m, "PyFuncResult");
  py_func_result
    .def(pybind11::init<>())
    .def_readwrite("status", &PyFuncResult::status)
    .def_readwrite("result", &PyFuncResult::result);

  pybind11::class_<PyFuncResult::TensorBuffer>(py_func_result, "TensorBuffer")
    .def(pybind11::init<>())
    .def_readwrite("buf", &PyFuncResult::TensorBuffer::buf)
    .def_readwrite("shape", &PyFuncResult::TensorBuffer::shape)
    .def_readwrite("type", &PyFuncResult::TensorBuffer::type);

  pybind11::bind_vector<std::vector<PyFuncResult::TensorBuffer>>(
      py_func_result, "TensorBufferVector");
}

}  // namespace python_lib
}  // namespace xdl

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

#include "xdl/core/backend/tf/convert_utils.h"
#include "xdl/core/backend/device_singleton.h"

namespace xdl {

Status XDL2TF::ConvertType(const xdl::DataType s, tensorflow::DataType* d) {
  switch(s) {
  case DataType::kInt8: 
    *d = tensorflow::DT_INT8; 
    return Status::Ok();
  case DataType::kInt16: 
    *d = tensorflow::DT_INT16; 
    return Status::Ok();
  case DataType::kInt32: 
    *d = tensorflow::DT_INT32; 
    return Status::Ok();
  case DataType::kInt64: 
    *d = tensorflow::DT_INT64; 
    return Status::Ok();
  case DataType::kFloat: 
    *d = tensorflow::DT_FLOAT; 
    return Status::Ok();
  case DataType::kDouble: 
    *d = tensorflow::DT_DOUBLE; 
    return Status::Ok();
  default:
    return Status::Internal("unsupported type:" + std::to_string(s));
  }
}

Status XDL2TF::ConvertShape(const xdl::TensorShape& s, tensorflow::TensorShape* d) {
  tensorflow::TensorShape shape;
  for (auto dim: s.Dims()) {
    shape.AddDim(dim);
  }

  *d = shape;
  return Status::Ok();
}

Status XDL2TF::ConvertTensor(const xdl::Tensor& s, tensorflow::Tensor* d) {
  tensorflow::DataType type;
  tensorflow::TensorShape shape;
  XDL_CHECK_STATUS(ConvertType(s.Type(), &type));
  XDL_CHECK_STATUS(ConvertShape(s.Shape(), &shape));
  *d = tensorflow::Tensor(type, shape);
  void* dst = (void*)d->tensor_data().data();
  size_t size = d->tensor_data().size();
  memcpy(dst, s.Raw<void>(), size);
  return Status::Ok();
}

Status TF2XDL::ConvertType(const tensorflow::DataType s, xdl::DataType* d) {
  switch(s) {
  case tensorflow::DT_INT8: 
    *d = DataType::kInt8; 
    return Status::Ok();
  case tensorflow::DT_INT16: 
    *d = DataType::kInt16; 
    return Status::Ok();
  case tensorflow::DT_INT32: 
    *d = DataType::kInt32; 
    return Status::Ok();
  case tensorflow::DT_INT64: 
    *d = DataType::kInt64; 
    return Status::Ok();
  case tensorflow::DT_FLOAT: 
    *d = DataType::kFloat; 
    return Status::Ok();
  case tensorflow::DT_DOUBLE: 
    *d = DataType::kDouble; 
    return Status::Ok();
  default:
    return Status::Internal("unsupported type:" + std::to_string(s));
  }
}

Status TF2XDL::ConvertShape(const tensorflow::TensorShape& s, xdl::TensorShape* d) {
  std::vector<size_t> dims;
  for (size_t i = 0; i < s.dims(); i++) {
    dims.push_back(s.dim_size(i));
  }

  *d = TensorShape(dims);
  return Status::Ok();
}

Status TF2XDL::ConvertTensor(const tensorflow::Tensor& s, xdl::Tensor* d) {
  DataType type;
  XDL_CHECK_STATUS(ConvertType(s.dtype(), &type));
  TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(s.shape(), &shape));
  *d = Tensor(DeviceSingleton::CpuInstance(), shape, type);
  void* src = (void*)s.tensor_data().data();
  size_t size = s.tensor_data().size();
  memcpy(d->Raw<int8_t>(), src, size);
  return Status::Ok();
}

} // namespace xdl

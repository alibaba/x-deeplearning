/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#include "xdl/core/ops/ps_ops/convert_utils.h"

#include "ps-plus/common/initializer/none_initializer.h"
#include "xdl/core/framework/graph_def.h"
#include "xdl/core/backend/device_singleton.h"
#include "xdl/core/framework/device_registry.h"

namespace xdl {

Status PS2XDL::ConvertType(ps::DataType src, xdl::DataType* dst) {
  switch (src) {
  case ps::DataType::kInt8: 
    *dst = xdl::DataType::kInt8;
    return Status::Ok();
  case ps::DataType::kInt16: 
    *dst = xdl::DataType::kInt16;
    return Status::Ok();
  case ps::DataType::kInt32: 
    *dst = xdl::DataType::kInt32;
    return Status::Ok();
  case ps::DataType::kInt64: 
    *dst = xdl::DataType::kInt64;
    return Status::Ok();
  case ps::DataType::kFloat: 
    *dst = xdl::DataType::kFloat;
    return Status::Ok();
  case ps::DataType::kDouble: 
    *dst = xdl::DataType::kDouble;
    return Status::Ok();
  default:
    return Status::Internal("no match xdl datatype for ps datatype:" +
                            std::to_string(src));
  }
}

Status PS2XDL::ConvertShape(const ps::TensorShape& s, xdl::TensorShape* d) {
  *d = xdl::TensorShape(s.Dims());
  return Status::Ok();
}

Status PS2XDL::ConvertTensor(const ps::Tensor& s, xdl::Tensor* d) {
  DataType type;
  XDL_CHECK_STATUS(ConvertType(s.Type(), &type));
  TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(s.Shape(), &shape));
  *d = xdl::Tensor(DeviceSingleton::CpuInstance(), shape, type);
  memcpy(d->Raw<void>(),
         s.Raw<void>(),
         SizeOfType(d->Type()) * shape.NumElements());
  return Status::Ok();
}

Status PS2XDL::ConvertTensorZC(ps::Tensor& s, xdl::Tensor* d) {
  DataType type;
  XDL_CHECK_STATUS(ConvertType(s.Type(), &type));
  TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(s.Shape(), &shape));
  s.SetOwnBuffer(false);
  *d = xdl::Tensor(DeviceSingleton::CpuInstance(), shape, type, s.Raw<void>(), true);
  return Status::Ok();
}

Status PS2XDL::ConvertTensorWithCtx(const ps::Tensor& src, 
                                    OpKernelContext* ctx, 
                                    int output_id) {
  xdl::TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(src.Shape(), &shape));
  xdl::Tensor output;
  XDL_CHECK_STATUS(ctx->AllocateOutput(output_id, shape, &output));
  xdl::DataType type;
  Status st = ConvertType(src.Type(), &type);
  if (!st.IsOk()) {
    return st;
  }
  if (type != output.Type()) {
    return Status::ArgumentError("Output " + std::to_string(output_id) + " Type Error");
  }
  memcpy(output.Raw<void>(), 
         src.Raw<void>(), 
         SizeOfType(output.Type()) * shape.NumElements());
  return Status::Ok();
}

xdl::Status PS2XDL::ConvertStatus(const ps::Status& st) {
  if (st.IsOk()) {
    return Status::Ok();
  } else if (st.Code() == ps::Status::ErrorCode::kArgumentError) {
    return Status::ArgumentError(st.Msg());
  } else if (st.Code() == ps::Status::ErrorCode::kIndexOverflow) {
    return Status::IndexOverflow(st.Msg());
  } else if (st.Code() == ps::Status::ErrorCode::kDataLoss ||
             st.Code() == ps::Status::ErrorCode::kNotReady ||
             st.Code() == ps::Status::ErrorCode::kNetworkError ||
             st.Code() == ps::Status::ErrorCode::kTimeout || 
             st.Code() == ps::Status::ErrorCode::kVersionMismatch ||
             st.Code() == ps::Status::ErrorCode::kConcurrentExecution ||
             st.Code() == ps::Status::ErrorCode::kServerSerializeFailed ||
             st.Code() == ps::Status::ErrorCode::kServerDeserializeFailed ||
             st.Code() == ps::Status::ErrorCode::kClientSerializeFailed ||
             st.Code() == ps::Status::ErrorCode::kClientDeserializeFailed) {
    return Status::PsError(st.Msg());
  } else {
    return Status::Internal(st.Msg());
  } 
}

Status XDL2PS::ConvertType(xdl::DataType src, ps::DataType* dst) {
  switch (src) {
  case xdl::DataType::kInt8:
    *dst = ps::DataType::kInt8;
    return Status::Ok();
  case xdl::DataType::kInt16:
    *dst = ps::DataType::kInt16;
    return Status::Ok();
  case xdl::DataType::kInt32:
    *dst = ps::DataType::kInt32;
    return Status::Ok();
  case xdl::DataType::kInt64:
    *dst = ps::DataType::kInt64;
    return Status::Ok();
  case xdl::DataType::kFloat:
    *dst = ps::DataType::kFloat;
    return Status::Ok();
  case xdl::DataType::kDouble:
    *dst = ps::DataType::kDouble;
    return Status::Ok();
  default:
    return Status::Internal("no match ps datatype for xdl datatype:" +
                            std::to_string(src));
  }
}

Status XDL2PS::ConvertShape(const xdl::TensorShape& s, ps::TensorShape* d) {
  *d = ps::TensorShape(s.Dims());
  return Status::Ok();
}

Status XDL2PS::ConvertTensor(const xdl::Tensor& s, ps::Tensor* d) {
  ps::DataType type;
  XDL_CHECK_STATUS(ConvertType(s.Type(), &type));
  ps::TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(s.Shape(), &shape));
  *d = ps::Tensor(type, shape, new ps::initializer::NoneInitializer());
  memcpy(d->Raw<void>(),
         s.Raw<void>(),
         SizeOfType(s.Type()) * shape.NumElements());
  return Status::Ok();
}

// zero copy version, by steal buffer
Status XDL2PS::ConvertTensorZC(xdl::Tensor& s, ps::Tensor* d) {
  ps::DataType type;
  XDL_CHECK_STATUS(ConvertType(s.Type(), &type));
  ps::TensorShape shape;
  XDL_CHECK_STATUS(ConvertShape(s.Shape(), &shape));
  *d = ps::Tensor(type, shape, s.Raw<char>(), new ps::initializer::NoneInitializer());
  return Status::Ok();
}

ps::Status XDL2PS::ConvertStatus(const Status& st) {
  if (st.IsOk()) {
    return ps::Status::Ok();
  } else if (st.Code() == Status::ErrorCode::kArgumentError) {
    return ps::Status::ArgumentError(st.Msg());
  } else if (st.Code() == Status::ErrorCode::kIndexOverflow) {
    return ps::Status::IndexOverflow(st.Msg());
  } else if (st.Code() == Status::ErrorCode::kPsError) {
    return ps::Status::Timeout(st.Msg());
  } else {
    return ps::Status::Unknown(st.Msg());
  }
}

}  // namespace xdl

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

namespace xdl {

template <typename T>
class ConstantOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    TensorShape shape;
    std::string value;
    XDL_CHECK_STATUS(ctx->GetAttr("shape", &shape));
    XDL_CHECK_STATUS(ctx->GetAttr("value", &value));
    XDL_CHECK_COND(shape.NumElements() * sizeof(T) == value.size(),
                   Status::ArgumentError("shape is not equal to value"));
    tensor_ = Tensor(ctx->GetDevice(), shape, DataTypeToEnum<T>::v());
    memcpy(tensor_.Raw<T>(), value.data(), value.size());
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    XDL_CHECK_STATUS(ctx->SetOutput(0, tensor_));
    return Status::Ok();
  }
 private:
  Tensor tensor_;
};

XDL_DEFINE_OP(_Constant)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("shape", AttrValue::kTensorShape)
  .Attr("value", AttrValue::kString)
  .Output("output", "dtype");

#define REGISTER_CPU_KERNEL(T)                 \
  XDL_REGISTER_KERNEL(_Constant, ConstantOp<T>) \
    .Device("CPU")                             \
    .AttrDataType<T>("dtype");

REGISTER_CPU_KERNEL(bool)
REGISTER_CPU_KERNEL(int8_t)
REGISTER_CPU_KERNEL(int16_t)
REGISTER_CPU_KERNEL(int32_t)
REGISTER_CPU_KERNEL(int64_t)
REGISTER_CPU_KERNEL(float)
REGISTER_CPU_KERNEL(double)

#undef REGISTER_CPU_KERNEL

}  // namespace xdl

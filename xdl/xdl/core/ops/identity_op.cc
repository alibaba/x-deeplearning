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
class IdentityOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor input;
    XDL_CHECK_STATUS(ctx->GetInput(0, &input));    
    Tensor output;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, input.Shape(), &output));
    memcpy(output.Raw<char>(), 
           input.Raw<char>(), 
           output.Shape().NumElements() * SizeOfType(output.Type()));
    return Status::Ok();
  }
};

XDL_DEFINE_OP(IdentityOp)
  .Input("input", "dtype")
  .Output("output", "dtype")
  .Attr("dtype", AttrValue::kDataType);

#define REGISTER_CPU_KERNEL(T)                      \
  XDL_REGISTER_KERNEL(IdentityOp, IdentityOp<T>)    \
    .Device("CPU")                                    \
    .AttrDataType<T>("dtype");

REGISTER_CPU_KERNEL(int8_t)
REGISTER_CPU_KERNEL(int16_t)
REGISTER_CPU_KERNEL(int32_t)
REGISTER_CPU_KERNEL(int64_t)
REGISTER_CPU_KERNEL(float)
REGISTER_CPU_KERNEL(double)

}  // namespace xdl



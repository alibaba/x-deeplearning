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

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"

namespace xdl {

class ZerosOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("dtype", &dtype_));
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor shape;
    XDL_CHECK_STATUS(ctx->GetInput(0, &shape));
    if (shape.Shape().Size() != 1) {
        return Status::ArgumentError("Zeros shape should be 1-D");
    }
    std::vector<size_t> shapex;
    int64_t num_ele = 1, loc = -1;
    for (int64_t i = 0; i < shape.Shape()[0]; i++) {
      int64_t x = shape.Raw<int64_t>()[i];
      if (x < 0) {
        return Status::ArgumentError("Zeros shape[i] < 0");
      }
      shapex.push_back(x);
    }
    Tensor output;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape(shapex), &output));
    memset(output.Raw<void>(), 0, SizeOfType(dtype_) * output.Shape().NumElements());
    return Status::Ok();
  }
 private:
  DataType dtype_;
};

XDL_DEFINE_OP(Zeros)
  .Input("shape", DataType::kInt64)
  .Attr("dtype", AttrValue::kDataType)
  .Output("result", "dtype");

XDL_REGISTER_KERNEL(Zeros, ZerosOp)
  .Device("CPU");

} // namespace xdl


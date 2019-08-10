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

class ReshapeOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("dtype", &dtype_));
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor tensor, shape;
    XDL_CHECK_STATUS(ctx->GetInput(0, &tensor));
    XDL_CHECK_STATUS(ctx->GetInput(1, &shape));
    if (shape.Shape().Size() != 1) {
        return Status::ArgumentError("Reshape shape should be 1-D");
    }
    std::vector<size_t> shapex;
    int64_t num_ele = 1, loc = -1;
    for (int64_t i = 0; i < shape.Shape()[0]; i++) {
      int64_t x = shape.Raw<int64_t>()[i];
      if (x < -1) {
        return Status::ArgumentError("Reshape shape[i] < -1");
      }
      if (x == -1) {
        shapex.push_back(0);
        if (loc != -1) {
          return Status::ArgumentError("Reshape shape has only 1 '-1'");
        }
        loc = i;
      } else {
        num_ele *= x;
        shapex.push_back(x);
      }
    }
    if (loc == -1) {
      if (num_ele != tensor.Shape().NumElements()) {
        return Status::ArgumentError("Reshape shape Mismatch");
      }
    } else {
      if (num_ele == 0) {
        return Status::ArgumentError("Reshape shape cannot have 0 and -1");
      }
      if (tensor.Shape().NumElements() % num_ele != 0) {
        return Status::ArgumentError("Reshape shape Mismatch, cannot be divided");
      }
      shapex[loc] = tensor.Shape().NumElements() / num_ele;
    }
    XDL_CHECK_STATUS(ctx->SetOutput(0, Tensor(TensorShape(shapex), dtype_, tensor.GetBuffer())));
    return Status::Ok();
  }
 private:
  DataType dtype_;
};

XDL_DEFINE_OP(Reshape)
  .Input("tensor", "dtype")
  .Input("shape", DataType::kInt64)
  .Attr("dtype", AttrValue::kDataType)
  .Output("result", "dtype");

XDL_REGISTER_KERNEL(Reshape, ReshapeOp)
  .Device("CPU");

} // namespace xdl


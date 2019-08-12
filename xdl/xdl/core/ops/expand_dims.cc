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

class ExpandDimsOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("axis", &axis_));
    XDL_CHECK_STATUS(ctx->GetAttr("dtype", &dtype_));
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor tensor;
    XDL_CHECK_STATUS(ctx->GetInput(0, &tensor));
    std::vector<size_t> shape = tensor.Shape().Dims();
    int64_t real_axis;
    if (axis_ < 0) {
      if (-axis_ > shape.size() + 1) {
        return Status::ArgumentError("ExpandDims axis_ should be [-value_dims, value_dims)");
      }
      real_axis = shape.size() + axis_ + 1;
    } else {
      if (axis_ >= shape.size() + 1) {
        return Status::ArgumentError("ExpandDims axis_ should be [-value_dims, value_dims)");
      }
      real_axis = axis_;
    }
    shape.insert(shape.begin() + real_axis, 1);
    XDL_CHECK_STATUS(ctx->SetOutput(0, Tensor(TensorShape(shape), dtype_, tensor.GetBuffer())));
    return Status::Ok();
  }
 private:
  int64_t axis_;
  DataType dtype_;
};

XDL_DEFINE_OP(ExpandDims)
  .Input("tensor", "dtype")
  .Attr("axis", AttrValue::kInt)
  .Attr("dtype", AttrValue::kDataType)
  .Output("result", "dtype");

XDL_REGISTER_KERNEL(ExpandDims, ExpandDimsOp)
  .Device("CPU");

} // namespace xdl


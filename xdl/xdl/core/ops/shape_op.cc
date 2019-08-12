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

#include "xdl/core/utils/logging.h"

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include <sstream>

namespace xdl {

class ShapeOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor input;
    XDL_CHECK_STATUS(ctx->GetInput(0, &input));
    Tensor shape;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape({input.Shape().Size()}), &shape));    
    int64_t* ptr = shape.Raw<int64_t>();
    for (size_t i = 0; i < input.Shape().Size(); ++i) {
      *ptr++ = input.Shape()[i];
    }

    return Status::Ok();
  }
};

XDL_DEFINE_OP(ShapeOp)
  .Input("input", "dtype")
  .Output("shape", DataType::kInt64)
  .Attr("dtype", AttrValue::kDataType);

XDL_REGISTER_KERNEL(ShapeOp, ShapeOp)
  .Device("CPU")
  .AttrDataType<int8_t>("dtype");
XDL_REGISTER_KERNEL(ShapeOp, ShapeOp)
  .Device("CPU")
  .AttrDataType<int16_t>("dtype");
XDL_REGISTER_KERNEL(ShapeOp, ShapeOp)
  .Device("CPU")
  .AttrDataType<int32_t>("dtype");
XDL_REGISTER_KERNEL(ShapeOp, ShapeOp)
  .Device("CPU")
  .AttrDataType<int64_t>("dtype");
XDL_REGISTER_KERNEL(ShapeOp, ShapeOp)
  .Device("CPU")
  .AttrDataType<float>("dtype");
XDL_REGISTER_KERNEL(ShapeOp, ShapeOp)
  .Device("CPU")
  .AttrDataType<double>("dtype");

} // namespace xdl



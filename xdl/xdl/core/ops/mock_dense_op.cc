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

namespace xdl {

class MockDenseOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("shape", &shape_));
    XDL_CHECK_STATUS(ctx->GetAttr("value", &value_));
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor output;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, shape_, &output));
    float* output_ptr = output.Raw<float>();
    for (size_t i = 0; i < shape_.NumElements(); ++i) {
      output_ptr[i] = value_;
    }

    return Status::Ok();
  }

 private:
  TensorShape shape_;
  float value_;
};

XDL_DEFINE_OP(MockDenseOp)
  .Output("output", DataType::kFloat)
  .Attr("shape", AttrValue::kTensorShape)
  .Attr("value", AttrValue::kFloat);

XDL_REGISTER_KERNEL(MockDenseOp, MockDenseOp).Device("CPU");

} // namespace xdl



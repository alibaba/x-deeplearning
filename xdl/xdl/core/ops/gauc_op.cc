/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "xdl/core/utils/logging.h"

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"

namespace xdl {

class GaucOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }
  
  Status Compute(OpKernelContext* ctx) override {
    Tensor auc, pv_num, output;
    XDL_CHECK_STATUS(ctx->GetInput(0, &auc));
    XDL_CHECK_STATUS(ctx->GetInput(1, &pv_num));
    double auc_value = *(auc.Raw<double>());
    int64_t num = *(pv_num.Raw<int64_t>());
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape({}), &output));
    if (num == 0) {
      XDL_LOG(INFO) << "pv_num is 0, cannot calculate gauc";
      *(output.Raw<double>()) = 0;
    } else {
      *(output.Raw<double>()) = auc_value / num;
    }
    return Status::Ok();
  }
};

XDL_DEFINE_OP(GaucOp)
  .Input("auc_value", DataType::kDouble)
  .Input("pv_num", DataType::kInt64)
  .Output("gauc", DataType::kDouble);

XDL_REGISTER_KERNEL(GaucOp, GaucOp).Device("CPU");

}  // namespace xdl

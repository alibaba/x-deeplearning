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

#include <math.h>

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"

namespace xdl {

class AucOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor tp;
    XDL_CHECK_STATUS(ctx->GetInput(0, &tp));    
    Tensor fp;
    XDL_CHECK_STATUS(ctx->GetInput(1, &fp));    
    Tensor tn;
    XDL_CHECK_STATUS(ctx->GetInput(2, &tn));    
    Tensor fn;
    XDL_CHECK_STATUS(ctx->GetInput(3, &fn));    
    Tensor auc;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape({}), &auc));
    *(auc.Raw<float>()) = ComputeAuc(tp, fp, tn, fn);
    return Status::Ok();
  }

  float ComputeAuc(const Tensor& tp_tensor,
                   const Tensor& fp_tensor,
                   const Tensor& tn_tensor,
                   const Tensor& fn_tensor) {
    int64_t* tp = tp_tensor.Raw<int64_t>();
    int64_t* fp = fp_tensor.Raw<int64_t>();
    int64_t* tn = tn_tensor.Raw<int64_t>();
    int64_t* fn = fn_tensor.Raw<int64_t>();
    std::vector<float> tpr;
    std::vector<float> fpr;
    int64_t num_thresholds = tp_tensor.Shape()[0];
    tpr.reserve(num_thresholds);
    fpr.reserve(num_thresholds);
    for (int64_t i = 0; i < num_thresholds; ++i) {
      tpr[i] = (tp[i] + kEpsilon) / (tp[i] + fn[i] + kEpsilon);
      fpr[i] = fp[i] / (fp[i] + tn[i] + kEpsilon);
    }

    float auc = 0;
    for (size_t i = 0; i < num_thresholds - 1; ++i) {
      auc += (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2.0;
    }

    return fabs(auc);
  }

 private:
  static constexpr float kEpsilon = 1e-6;
};

XDL_DEFINE_OP(AucOp)
  .Input("tp", DataType::kInt64)
  .Input("fp", DataType::kInt64)
  .Input("tn", DataType::kInt64)
  .Input("fn", DataType::kInt64)
  .Output("auc", DataType::kFloat);

XDL_REGISTER_KERNEL(AucOp, AucOp).Device("CPU");

}  // namespace xdl



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

class SimpleKsumOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor emb;
    XDL_CHECK_STATUS(ctx->GetInput(0, &emb));
    XDL_CHECK_COND(
        2 == emb.Shape().Size(), 
        Status::ArgumentError("embedding input dim must be 2"));
    Tensor idx;
    XDL_CHECK_STATUS(ctx->GetInput(1, &idx));
    XDL_CHECK_COND(
        1 == idx.Shape().Size(), 
        Status::ArgumentError("idx dim must be 1"));
    Tensor segments;
    XDL_CHECK_STATUS(ctx->GetInput(2, &segments));
    Tensor output;
    size_t emb_dim = emb.Shape()[1];
    size_t batch_size = segments.Shape()[0];
    XDL_CHECK_STATUS(
        ctx->AllocateOutput(
            0, TensorShape({batch_size, emb_dim}), 
            &output));    
    int32_t* idx_ptr = idx.Raw<int32_t>();
    float* emb_ptr = emb.Raw<float>();
    int32_t* segment_id_ptr = segments.Raw<int32_t>();
    float* output_ptr = output.Raw<float>();
    bool first_run = true;
    for (size_t i = 0, segment_id = 0; i < idx.Shape()[0]; ++i) {
      float* src = emb_ptr + *(idx_ptr + i) * emb_dim;
      if (i >= *(segment_id_ptr + segment_id)) {
        segment_id++;
      }

      float* dst = output_ptr + emb_dim * segment_id;
      if (first_run) {
        first_run = false;
        memcpy(dst, src, emb_dim * sizeof(float));
      } else {
        for (size_t j = 0; j < emb_dim; ++j) {
          dst[j] += src[j];
        }
      }
    }

    return Status::Ok();
  }
};

XDL_DEFINE_OP(SimpleKsumOp)
  .Input("embeddings", DataType::kFloat)
  .Input("idx", DataType::kInt32)
  .Input("segments", DataType::kInt32)
  .Output("output", DataType::kFloat);

XDL_REGISTER_KERNEL(SimpleKsumOp, SimpleKsumOp).Device("CPU");

} // namespace xdl



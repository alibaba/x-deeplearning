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

class MfeatureToHash64 : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("default_value", &default_value_));
    XDL_CHECK_STATUS(ctx->GetAttr("pad", &pad_len_));
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor t_ids, t_segment, t_output;
    XDL_CHECK_STATUS(ctx->GetInput(0, &t_ids));
    XDL_CHECK_STATUS(ctx->GetInput(1, &t_segment));
    size_t batch_size = t_segment.Shape()[0];
    TensorShape output_shape{batch_size, pad_len_};
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, output_shape, &t_output));
    int32_t* segment = t_segment.Raw<int32_t>();
    int64_t* ids = t_ids.Raw<int64_t>();
    int64_t* output = t_output.Raw<int64_t>();
    for (size_t i = 0; i < batch_size; i++) {
      int32_t seg_size = i == 0 ? segment[i] : segment[i] - segment[i - 1];
      int32_t start = i == 0 ? 0 : segment[i - 1];
      for (size_t j = 0; j < pad_len_; j++) {
        if (j < seg_size) {
          output[i * pad_len_ + j] = ids[2 * (start + j)];
        } else {
          output[i * pad_len_ + j] = default_value_;
        }
      }
    }
    return Status::Ok();
  }
 private:
  int64_t default_value_;
  int64_t pad_len_;  
  DataType dtype_;
};

XDL_DEFINE_OP(MfeatureToHash64)
  .Input("ids", DataType::kInt64)
  .Input("segment", DataType::kInt32)
  .Attr("default_value", AttrValue::kInt)
  .Attr("pad", AttrValue::kInt)
  .Output("result", DataType::kInt64);

XDL_REGISTER_KERNEL(MfeatureToHash64, MfeatureToHash64)
  .Device("CPU");

} // namespace xdl


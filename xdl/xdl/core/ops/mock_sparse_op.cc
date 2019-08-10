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

#include <stdlib.h>
#include <time.h>

namespace xdl {

class MockSparseOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("dense_shape", &shape_));
    srand(time(NULL));
    return Status::Ok();
  }

  float RandomFloat() {
    return (float)(rand() % 100) / 100;
  }

  int64_t RandomInt(size_t max) {
    return rand() % max;
  }

  Status Compute(OpKernelContext* ctx) override {
    std::vector<size_t> id_cnts;
    size_t total_id_cnt = 0;
    for (size_t i = 0; i < shape_[0]; ++i) {
      int64_t id_cnt = RandomInt(shape_[1]);
      if (id_cnt == 0) id_cnt = 1;
      id_cnts.push_back(id_cnt);
      total_id_cnt += id_cnts.back();
    }

    Tensor ids;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape({total_id_cnt}), &ids));
    Tensor values;
    XDL_CHECK_STATUS(ctx->AllocateOutput(1, TensorShape({total_id_cnt}), &values));
    Tensor segments;
    XDL_CHECK_STATUS(ctx->AllocateOutput(2, TensorShape({shape_[0]}), &segments));

    int64_t* ids_ptr = ids.Raw<int64_t>();
    float* values_ptr = values.Raw<float>();    
    int32_t* segments_ptr = segments.Raw<int32_t>();

    for (size_t i = 0; i < total_id_cnt; ++i) {
      ids_ptr[i] = RandomInt(shape_[1]);
    }

    for (size_t i = 0; i < total_id_cnt; ++i) {
      values_ptr[i] = RandomFloat();
    }

    size_t sample_index = 0;
    for (size_t i = 0; i < shape_[0]; ++i) {
      sample_index += id_cnts[i];
      segments_ptr[i] = sample_index;
    }
    
    return Status::Ok();
  }

 private:
  TensorShape shape_;
  float value_;
};

XDL_DEFINE_OP(MockSparseOp)
  .Output("ids", DataType::kInt64)
  .Output("values", DataType::kFloat)
  .Output("segments", DataType::kInt32)
  .Attr("dense_shape", AttrValue::kTensorShape);

XDL_REGISTER_KERNEL(MockSparseOp, MockSparseOp).Device("CPU");

} // namespace xdl



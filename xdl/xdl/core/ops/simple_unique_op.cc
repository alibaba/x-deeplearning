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

template <typename T>
class SimpleUniqueOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor input;
    XDL_CHECK_STATUS(ctx->GetInput(0, &input));
    T* input_ptr = input.Raw<T>();
    size_t N = input.Shape()[0];
    Tensor idx;
    XDL_CHECK_STATUS(ctx->AllocateOutput(1, TensorShape({N}), &idx));
    int32_t* idx_ptr = idx.Raw<int32_t>();
    std::unordered_map<T, int32_t> uniq;
    uniq.reserve(N);
    std::stringstream ss;
    for (int64_t i = 0, j = 0; i < N; i++) {
      auto it = uniq.insert({input_ptr[i], j});
      if (it.second) {
        ++j;
      }

      idx_ptr[i] = it.first->second;
    }

    size_t uniq_size = uniq.size();    
    Tensor output;
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape({uniq_size}), &output));    
    T* output_ptr = output.Raw<T>();
    for (auto it : uniq) {
      output_ptr[it.second] = it.first;
    }

    return Status::Ok();
  }
};

XDL_DEFINE_OP(SimpleUniqueOp)
  .Input("x", "dtype")
  .Output("y", "dtype")
  .Output("idx", DataType::kInt32)
  .Attr("dtype", AttrValue::kDataType);

XDL_REGISTER_KERNEL(SimpleUniqueOp, SimpleUniqueOp<int32_t>)
  .Device("CPU")
  .AttrDataType<int32_t>("dtype");

XDL_REGISTER_KERNEL(SimpleUniqueOp, SimpleUniqueOp<int64_t>)
  .Device("CPU")
  .AttrDataType<int64_t>("dtype");

} // namespace xdl



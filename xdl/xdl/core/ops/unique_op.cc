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

#include "xdl/core/lib/unique.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"

#include <chrono>
#include <vector>
#include <utility>

namespace xdl {

template <typename T, typename I>
class UniqueCpuOp : public OpKernel {
 public:
  Status Compute(OpKernelContext* ctx) override;
};

template <typename T, typename I>
Status UniqueCpuOp<T, I>::Compute(OpKernelContext* ctx) {
  //auto t0 = std::chrono::high_resolution_clock::now();
  Tensor input, segment, output, out_index, sample_index, sample_segment;
  XDL_CHECK_STATUS(ctx->GetInput(0, &input));
  XDL_CHECK_STATUS(ctx->GetInput(1, &segment));
  XDL_CHECK_COND(2 >= input.Shape().Size(),
                 Status::ArgumentError("input dim cann't be greater than 2"));

  CpuDevice* device = dynamic_cast<CpuDevice*>(ctx->GetDevice());
  auto fn = functor::UniqueFunctor<CpuDevice, T, I>();
  fn(device, input, segment, &output, &out_index, &sample_index, &sample_segment);

  ctx->SetOutput(0, output);
  ctx->SetOutput(1, out_index);  
  ctx->SetOutput(2, sample_index);
  ctx->SetOutput(3, sample_segment);  
  //auto t1 = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double, std::milli> diff = t1 - t0;
  //LOG(INFO) << "cpu unique op time:" << diff.count() << "ms";
  return Status::Ok();
}

XDL_DEFINE_OP(Unique)
  .Input("input", "dtype")
  .Input("segment", "itype")
  .Output("output", "dtype")
  .Output("index", "itype")
  .Output("sample_index", "itype")
  .Output("sample_segment", "itype")
  .Attr("dtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType);

#define REGISTER_KERNEL(T, I)                    \
  XDL_REGISTER_KERNEL(Unique, UniqueCpuOp<T, I>) \
    .Device("CPU")                               \
    .AttrDataType<T>("dtype")                    \
    .AttrDataType<I>("itype")

REGISTER_KERNEL(int64_t, int64_t);
REGISTER_KERNEL(int32_t, int32_t);
REGISTER_KERNEL(int64_t, int32_t);
REGISTER_KERNEL(int32_t, int64_t);

#undef REGISTER_KERNEL

}  // namespace xdl

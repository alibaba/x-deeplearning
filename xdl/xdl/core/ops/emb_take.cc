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
#include "xdl/core/ops/take_op.h"

#include <omp.h>
#include <vector>

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/utils/logging.h"


namespace xdl {

template <typename T, typename I>
class EmbTakeOp : public OpKernel {
 public:
  Status Compute(OpKernelContext* ctx) override {
    Tensor input, seg, rst;
    XDL_CHECK_STATUS(ctx->GetInput(0, &input));
    XDL_CHECK_STATUS(ctx->GetInput(1, &seg));
    XDL_CHECK_COND(seg.Shape().Size() == 1,
                   Status::ArgumentError("seg Shape should be rank-1"));
    auto dims = input.Shape().Dims();
    size_t size = SizeOfType(input.Type()) * input.Shape().NumElements() / dims[0];
    dims[0] = seg.Shape().NumElements();
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape(dims), &rst));
    I old = 0;
    I* ids = seg.Raw<I>();
    for (auto i = 0; i < seg.Shape().NumElements(); i++) {
      if (old == ids[i]) {
        memset(rst.Raw<char>() + i * size, 0, size);
      } else {
        memcpy(rst.Raw<char>() + i * size, input.Raw<char>() + old * size, size);
        old = ids[i];
      }
    }
    return Status::Ok();
  }
};

XDL_DEFINE_OP(EmbTakeOp)
  .Input("data", "dtype")
  .Input("seg", "itype")
  .Output("output", "dtype")
  .Attr("dtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType);

#define REGISTER_KERNEL(T, I)               \
  XDL_REGISTER_KERNEL(EmbTakeOp, EmbTakeOp<T, I>) \
  .Device("CPU")                            \
  .AttrDataType<T>("dtype")                 \
  .AttrDataType<I>("itype")

REGISTER_KERNEL(int32_t, int32_t);
REGISTER_KERNEL(int32_t, int64_t);
REGISTER_KERNEL(int64_t, int32_t);
REGISTER_KERNEL(int64_t, int64_t);

#undef REGISTER_KERNEL

}  // namespace xdl

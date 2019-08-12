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

#include "xdl/core/ops/take_grad_op.h"

#include <omp.h>

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/atomic.h"

namespace xdl {

template <typename T, typename I>
Status TakeGradOp<T, I>::Compute(OpKernelContext* ctx) {
  Tensor grad, indicator, feature, output;
  XDL_CHECK_STATUS(ctx->GetInput(0, &grad));
  XDL_CHECK_STATUS(ctx->GetInput(1, &indicator));
  XDL_CHECK_STATUS(ctx->GetInput(2, &feature));
  XDL_CHECK_COND(1 == indicator.Shape().Size(),
                 Status::ArgumentError("indicator must be rank 1 tensor"));
  XDL_CHECK_COND(grad.Shape()[0] == indicator.Shape().NumElements(),
                 Status::ArgumentError("grad and indicator size not match"));

  auto grad_dims = grad.Shape().Dims();
  size_t row = grad_dims[0];
  size_t col = grad.Shape().NumElements() / row;
  T* pin = grad.Raw<T>();
  I* pind = indicator.Raw<I>();
  std::vector<size_t> dims(grad_dims.begin(), grad_dims.end());
  dims[0] = *(feature.Raw<int64_t>());
  TensorShape out_shape(dims);
  XDL_CHECK_STATUS(ctx->AllocateOutput(0, out_shape, &output));
  T* pout = output.Raw<T>();
  std::memset(pout, 0, sizeof(T) * out_shape.NumElements());
  
  #pragma omp parallel for
  for (size_t k = 0; k < row * col; ++k) {
    size_t i = k / col;
    size_t j = k % col;
    I rrow = pind[i];
    common::cpu_atomic_add<T>(pin[k], pout + rrow * col + j);
  }
  return Status::Ok();
}

XDL_DEFINE_OP(TakeGrad)
  .Input("grad", "dtype")
  .Input("indicator", "itype")
  .Input("feature", DataType::kInt64)
  .Output("output", "dtype")
  .Attr("dtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType);

#define REGISTER_KERNEL(T, I)                     \
  XDL_REGISTER_KERNEL(TakeGrad, TakeGradOp<T, I>) \
  .Device("CPU")                                  \
  .AttrDataType<T>("dtype")                       \
  .AttrDataType<I>("itype");

REGISTER_KERNEL(float, int32_t);
REGISTER_KERNEL(float, int64_t);
REGISTER_KERNEL(double, int32_t);
REGISTER_KERNEL(double, int64_t);

#undef REGISTER_KERNEL

}  // namespace xdl

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
Status TakeOp<T, I>::Init(OpKernelConstruction* ctx) {
  return Status::Ok();
}

template <typename T, typename I>
Status TakeOp<T, I>::Compute(OpKernelContext* ctx) {
  Tensor feature, indicator, output;
  XDL_CHECK_STATUS(ctx->GetInput(0, &feature));
  XDL_CHECK_STATUS(ctx->GetInput(1, &indicator));
  XDL_CHECK_COND(1 == indicator.Shape().Size(),
                 Status::ArgumentError("indicator must be rank 1 tensor"));

  auto fea_dims = feature.Shape().Dims();
  std::vector<size_t> dims(fea_dims.begin(), fea_dims.end());
  dims[0] = indicator.Shape().NumElements();
  TensorShape out_shape(dims);
  XDL_CHECK_STATUS(ctx->AllocateOutput(0, out_shape, &output));

  size_t row = dims[0];
  size_t col = feature.Shape().NumElements() / feature.Shape()[0];
  T* pin = feature.Raw<T>(), *pout = output.Raw<T>();
  I* pind = indicator.Raw<I>();
  if (row == 0 || col == 0) {
    std::memset(pout, 0, sizeof(T) * out_shape.NumElements());
    return Status::Ok();
  }

  #pragma omp parallel for
  for (size_t i = 0; i < row; ++i) {
    const size_t a = i * col, b = pind[i] * col;
    for (size_t j = 0; j < col; ++j) {
      pout[a + j] = pin[b + j];
    }
  }
  return Status::Ok();
}

XDL_DEFINE_OP(TakeOp)
  .Input("feature", "dtype")
  .Input("indicator", "itype")
  .Output("output", "dtype")
  .Attr("dtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType);

#define REGISTER_KERNEL(T, I)               \
  XDL_REGISTER_KERNEL(TakeOp, TakeOp<T, I>) \
  .Device("CPU")                            \
  .AttrDataType<T>("dtype")                 \
  .AttrDataType<I>("itype")

REGISTER_KERNEL(float, int32_t);
REGISTER_KERNEL(float, int64_t);
REGISTER_KERNEL(double, int32_t);
REGISTER_KERNEL(double, int64_t);
REGISTER_KERNEL(int64_t, int32_t);
REGISTER_KERNEL(int64_t, int64_t);

#undef REGISTER_KERNEL

}  // namespace xdl

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

#include "xdl/core/ops/ksum_op.h"

#include <omp.h>
#include <cstring>
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/framework/cpu_device.h"
#include "xdl/core/lib/atomic.h"
#include "xdl/core/utils/logging.h"

namespace xdl {

template <typename T, typename I>
Status KSumOp<T, I>::Init(OpKernelConstruction* ctx) {
  XDL_CHECK_STATUS(ctx->GetAttr("average", &average_));
  return Status::Ok();
}

template <typename T, typename I>
Status KSumOp<T, I>::Compute(OpKernelContext* ctx) {
  Tensor embed, index, value, segment, group, output;
  XDL_CHECK_STATUS(ctx->GetInput(0, &embed));
  XDL_CHECK_COND(2 == embed.Shape().Size(),
                 Status::ArgumentError("embed input dim must be 2"));
  XDL_CHECK_STATUS(ctx->GetInput(1, &index));
  XDL_CHECK_COND(1 == index.Shape().Size(),
                 Status::ArgumentError("index input dim must be 1"));
  XDL_CHECK_STATUS(ctx->GetInput(2, &value));
  XDL_CHECK_COND(value.Shape().NumElements() == index.Shape().NumElements() ||
                 value.Shape().NumElements() == 0,
                 Status::ArgumentError("value input size must match index"));
  XDL_CHECK_STATUS(ctx->GetInput(3, &segment));
  XDL_CHECK_COND(1 == segment.Shape().Size(),
                 Status::ArgumentError("segment input dim must be 1"));
  XDL_CHECK_STATUS(ctx->GetInput(4, &group));
  XDL_CHECK_COND(1 == group.Shape().Size(),
                 Status::ArgumentError("group input dim must be 1"));
  
  T* peb = embed.Raw<T>();
  I* pidx = index.Raw<I>();
  T* pval = value.Raw<T>();
  I* pseg = segment.Raw<I>();
  I* pgrp = group.Raw<I>();

  if (value.Shape().NumElements() == 0) {
    pval = nullptr;
  }

  size_t eb_dim = embed.Shape()[1];
  size_t seg_size = segment.Shape().NumElements();
  size_t id_size = index.Shape().NumElements();
  size_t grp_size = seg_size;

  TensorShape out_shape({seg_size, eb_dim});
  if (group.Shape().NumElements() == 0) {
    pgrp = pseg;
  } else {
    grp_size = group.Shape().NumElements();
    XDL_CHECK(grp_size % seg_size == 0) << "group must be divided by segment";
    size_t grp_num = grp_size / seg_size;
    out_shape.Set(1, out_shape[1] * grp_num);
  }
  XDL_CHECK_STATUS(ctx->AllocateOutput(0, out_shape, &output));
  T* pout = output.Raw<T>();
  std::memset(pout, 0, sizeof(T) * out_shape.NumElements());

  std::function<void(size_t)> func = [=] (size_t sample_id) {
    size_t beg = sample_id == 0 ? 0 : pgrp[sample_id - 1];
    size_t end = pgrp[sample_id];
    T* dst = pout + sample_id * eb_dim;
    for (size_t i = beg; i < end; ++i) {
      const T* src = peb + pidx[i] * eb_dim;
      for (size_t k = 0; k < eb_dim; ++k) {
        T val = (pval != nullptr) ? pval[i] * src[k] : src[k];
        if (average_) val /= (end - beg);
        dst[k] += val;
      }
    }
  };
  if (grp_size < 100) {
    for (size_t sample_id = 0; sample_id < grp_size; ++sample_id) {
      func(sample_id);
    }
  } else {
    #pragma omp parallel for
    for (size_t sample_id = 0; sample_id < grp_size; ++sample_id) {
      func(sample_id);
    }
  }

  return Status::Ok();
}

XDL_DEFINE_OP(KSum)
  .Input("embed", "dtype")
  .Input("index", "itype")
  .Input("value", "dtype")
  .Input("segment", "itype")
  .Input("group", "itype")
  .Input("sample_index", "itype")
  .Input("sample_segment", "itype")
  .Output("output", "dtype")
  .Attr("dtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType)
  .Attr("average", AttrValue::kBool, false);

#define REGISTER_KERNEL(T, I)                \
  XDL_REGISTER_KERNEL(KSum, KSumOp<T, I>) \
  .Device("CPU")                             \
  .AttrDataType<T>("dtype")                  \
  .AttrDataType<I>("itype")

REGISTER_KERNEL(float, int32_t);
REGISTER_KERNEL(float, int64_t);
REGISTER_KERNEL(double, int32_t);
REGISTER_KERNEL(double, int64_t);
 
#undef REGISTER_KERNEL

}  // namespace xdl

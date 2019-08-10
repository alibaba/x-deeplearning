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

#include "xdl/core/ops/tile_grad_op.h"

#include <omp.h>
#include <cstring>
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/atomic.h"

namespace xdl {

template <typename T, typename I>
Status TileGradOp<T, I>::Init(OpKernelConstruction* ctx) {
  XDL_CHECK_STATUS(ctx->GetAttr("reverse", &reverse_));
  XDL_CHECK_STATUS(ctx->GetAttr("length", &length_));
  return Status::Ok();
}

template <typename T, typename I>
Status TileGradOp<T, I>::Compute(OpKernelContext* ctx) {
  Tensor embed, index, value, segment, group, grad, out_grad;
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
  XDL_CHECK_STATUS(ctx->GetInput(5, &grad));
  XDL_CHECK_COND(2 == grad.Shape().Size(),
                 Status::ArgumentError("grad input dim must be 2"));

  T* pgrad = grad.Raw<T>();
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
  XDL_CHECK(seg_size == grad.Shape()[0]) << "grad dim 0 is not equal to batch size";

  if (group.Shape().NumElements() == 0) {
    pgrp = pseg;
  } else {
    grp_size = group.Shape().NumElements();
    XDL_CHECK(grp_size % seg_size == 0) << "group must be divided by segment";
  }

  XDL_CHECK_STATUS(ctx->AllocateOutput(0, embed.Shape(), &out_grad));
  T* pout = out_grad.Raw<T>();
  std::memset(pout, 0, sizeof(T) * embed.Shape().NumElements());

  //#pragma omp parallel for
  for (size_t i = 0; i < id_size; ++i) {
    size_t grp_idx = std::lower_bound(pgrp, pgrp + grp_size, i + 1) - pgrp;
    size_t src_begin = grp_idx * length_;
    size_t dst_begin = pidx[i] * eb_dim;
    size_t grp_width = grp_idx == 0 ? pgrp[grp_idx]
                                    : pgrp[grp_idx] - pgrp[grp_idx - 1];
    if (grp_width == 0) continue;
    size_t grp_off = i - ((grp_idx == 0) ? 0 : pgrp[grp_idx - 1]);
    grp_off = reverse_ ? (grp_width - 1 - grp_off) : grp_off;
    for (size_t k = 0; k < eb_dim; ++k) {
      size_t off = grp_off * eb_dim + k;
      if (off < length_) {
        if (pval != nullptr) {
          common::cpu_atomic_add<T>(pval[i] * pgrad[src_begin + off],
                                    pout + dst_begin + k);
        } else {
          common::cpu_atomic_add<T>(pgrad[src_begin + off],
                                    pout + dst_begin + k);
        }
      }
    }
  }
  return Status::Ok();
}

XDL_DEFINE_OP(TileGrad)
    .Input("embed", "dtype")
    .Input("index", "itype")
    .Input("value", "dtype")
    .Input("segment", "itype")
    .Input("group", "itype")
    .Input("grad", "dtype")
    .Output("out_grad", "dtype")
    .Attr("dtype", AttrValue::kDataType)
    .Attr("itype", AttrValue::kDataType)
    .Attr("reverse", AttrValue::kBool, false)
    .Attr("length", AttrValue::kInt);

#define REGISTER_KERNEL(T, I)                     \
  XDL_REGISTER_KERNEL(TileGrad, TileGradOp<T, I>) \
  .Device("CPU")                                  \
  .AttrDataType<T>("dtype")                       \
  .AttrDataType<I>("itype")

REGISTER_KERNEL(float, int32_t);
REGISTER_KERNEL(float, int64_t);
REGISTER_KERNEL(double, int32_t);
REGISTER_KERNEL(double, int64_t);

#undef REGISTER_KERNEL

}  // namespace xdl

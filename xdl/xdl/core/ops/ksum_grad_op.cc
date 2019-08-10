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

#include "xdl/core/ops/ksum_grad_op.h"

#include <omp.h>
#include <cstring>
#include "xdl/core/utils/logging.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/framework/cpu_device.h"
#include "xdl/core/lib/atomic.h"
#include "ps-plus/ps-plus/common/thread_pool.h"

namespace xdl {

template <typename T, typename I>
Status KSumGradOp<T, I>::Init(OpKernelConstruction* ctx) {
  XDL_CHECK_STATUS(ctx->GetAttr("average", &average_));
  return Status::Ok();
}

template <typename T, typename I>
class Cal {
public:
  Cal(size_t eb_dim, size_t start, bool average, I* sidx, T* pgrad, T* dst, I* pgrp): eb_dim_(eb_dim), start_(start), average_(average), sidx_(sidx), pgrad_(pgrad), dst_(dst), pgrp_(pgrp) {}
  ps::Status operator()(const ps::Range& r) const {
    std::vector<T> tmp(eb_dim_, T());
    for (size_t j = start_ + r.begin; j < start_ + r.end; j++) {
      I idx = sidx_[j];
      for (size_t k = 0; k < eb_dim_; k++) {
        T val = pgrad_[idx*eb_dim_ + k];
        size_t grp_width = idx == 0 ? pgrp_[idx] : pgrp_[idx] - pgrp_[idx-1];
        if (average_) val /= grp_width;
        tmp[k] += val;
      }
    }
    for (size_t k = 0; k < eb_dim_; k++) {
      common::cpu_atomic_add<T>(tmp[k], dst_ + k);
    }
    return ps::Status::Ok();
  }
private:
  size_t eb_dim_;
  size_t start_;
  bool average_;
  I* sidx_;
  T* pgrad_;
  T* dst_;
  I* pgrp_;
};


template <typename T, typename I>
Status KSumGradOp<T, I>::Compute(OpKernelContext* ctx) {
  Tensor embed, index, value, segment, group, grad, sample_index, sample_segment, out_grad;
  XDL_CHECK_STATUS(ctx->GetInput(0, &embed));
  XDL_CHECK_COND(1 == embed.Shape().Size(),
                 Status::ArgumentError("embed input dim must be 1"));
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
  XDL_CHECK_STATUS(ctx->GetInput(5, &sample_index));
  XDL_CHECK_COND(1 == sample_index.Shape().Size(),
                 Status::ArgumentError("sample_index input dim must be 1"));
  XDL_CHECK_STATUS(ctx->GetInput(6, &sample_segment));
  XDL_CHECK_COND(1 == sample_segment.Shape().Size(),
                 Status::ArgumentError("sample_segment input dim must be 1"));  
  XDL_CHECK_STATUS(ctx->GetInput(7, &grad));
  XDL_CHECK_COND(2 == grad.Shape().Size(),
                 Status::ArgumentError("grad input dim must be 2"));

  T* pgrad = grad.Raw<T>();
  I* pidx = index.Raw<I>();
  T* pval = value.Raw<T>();
  I* pseg = segment.Raw<I>();
  I* pgrp = group.Raw<I>();
  I* sidx = sample_index.Raw<I>();
  I* sseg = sample_segment.Raw<I>();

  bool all_one = true;
  float FLOAT_EPSILON = 1.192092896e-07f;
  for (size_t i = 0; i < value.Shape().NumElements(); i++) {
    if ((pval[i] > (1 + FLOAT_EPSILON)) || (pval[i] < (1 - FLOAT_EPSILON))) {
      all_one = false;
      break;
    }
  }
  
  if (value.Shape().NumElements() == 0 || all_one) {
    pval = nullptr;
  }

  std::vector<size_t> dims;
  int64_t* ebs_ptr = embed.Raw<int64_t>();
  for (size_t i = 0; i < embed.Shape().NumElements(); ++i) {
    dims.push_back(*ebs_ptr++);
  }

  TensorShape embed_shape(dims);

  size_t eb_dim = embed_shape[1];
  size_t seg_size = segment.Shape().NumElements();
  size_t id_size = index.Shape().NumElements();
  size_t grp_size = seg_size;
  XDL_CHECK(seg_size == grad.Shape()[0]) << "grad dim 0 is not equal to batch size ";

  if (group.Shape().NumElements() == 0) {
    pgrp = pseg;
  } else {
    grp_size = group.Shape().NumElements();
    XDL_CHECK(grp_size % seg_size == 0) << "group must be divided by segment";
  }
  XDL_CHECK(grad.Shape()[1] % eb_dim == 0) << "grad shape[1] not equal to emb_dim";
  XDL_CHECK(sample_segment.Shape()[0] == embed_shape[0]) << "sample_segment size is not equal to eb_dim[0]";

  XDL_CHECK_STATUS(ctx->AllocateOutput(0, embed_shape, &out_grad));
  T* pout = out_grad.Raw<T>();
  std::memset(pout, 0, sizeof(T) * embed_shape.NumElements());
  if (pval != nullptr) {
    if (id_size < 1000) {
      auto func = [pgrad, pout, pidx, eb_dim, pval, this] (I i, size_t grp_idx, size_t grp_width, const T* src) {
        T* dst = pout + pidx[i] * eb_dim;
        for (size_t k = 0; k < eb_dim; ++k) {
          T val = (pval != nullptr) ? pval[i] * src[k] : src[k];
          if (this->average_) val /= grp_width;
          dst[k] += val;
        }
      };
      I i = 0;
      const I* p = pgrp;
      const I* q;
      for (size_t grp_width, grp_idx = 0; grp_idx < grp_size; ++grp_idx, q = p++) {
        grp_width = grp_idx == 0 ? *p : *p - *q;
        if (grp_width == 0) continue;
        const T* src = pgrad + grp_idx * eb_dim;
        for (; i < *p; ++i) {
          func(i, grp_idx, grp_width, src);
        }
      }
      return Status::Ok();
    }
  
    #pragma omp parallel for
    for(size_t i = 0; i < id_size; ++i) {
      size_t grp_idx = std::lower_bound(pgrp, pgrp + grp_size, i + 1) - pgrp;
      size_t grp_width = (grp_idx == 0) ? pgrp[grp_idx] : (pgrp[grp_idx] - pgrp[grp_idx - 1]);
      if (grp_width == 0) continue;
      const T* src = pgrad + grp_idx * eb_dim;
      T* dst = pout + pidx[i] * eb_dim;
      for (size_t k = 0; k < eb_dim; ++k) {
        T val = (pval != nullptr) ? pval[i] * src[k] : src[k];
        if (average_) val /= grp_width;
        common::cpu_atomic_add<T>(val, dst + k);
      }
    }
  } else {
    //auto time_start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for(size_t i = 0; i < sample_segment.Shape()[0]; i++) {
      size_t start = i == 0 ? 0 : sseg[i-1];
      size_t end = sseg[i];
      T* dst = pout + i * eb_dim;
      if (end-start > 300) {
        Cal<T,I> c(eb_dim, start, average_, sidx, pgrad, dst, pgrp);
        ps::MultiThreadDo(end-start, c, 300);
      } else {
        for (size_t j = start; j < end; j++) {
          I idx = sidx[j];
          for (size_t k = 0; k < eb_dim; k++) {
            T val = pgrad[idx*eb_dim + k];
            size_t grp_width = idx == 0 ? pgrp[idx] : pgrp[idx] - pgrp[idx-1];
            if (average_) val /= grp_width;
            dst[k] += val;
          }
        }
      }
    }
    //auto time_end = std::chrono::system_clock::now();
    //printf("ksum time %ld\n", (time_end-time_start).count());
  }
  return Status::Ok();
}

XDL_DEFINE_OP(KSumGrad)
  .Input("embed", DataType::kInt64)
  .Input("index", "itype")
  .Input("value", "dtype")
  .Input("segment", "itype")
  .Input("group", "itype")
  .Input("sample_index", "itype")
  .Input("sample_segment", "itype")
  .Input("grad", "dtype")
  .Output("out_grad", "dtype")
  .Attr("dtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType)
  .Attr("average", AttrValue::kBool, false);

#define REGISTER_KERNEL(T, I)                     \
  XDL_REGISTER_KERNEL(KSumGrad, KSumGradOp<T, I>) \
  .Device("CPU")                                  \
  .AttrDataType<T>("dtype")                       \
  .AttrDataType<I>("itype")

REGISTER_KERNEL(float, int32_t);
REGISTER_KERNEL(float, int64_t);
REGISTER_KERNEL(double, int32_t);
REGISTER_KERNEL(double, int64_t);
 
#undef REGISTER_KERNEL

}  // namespace xdl

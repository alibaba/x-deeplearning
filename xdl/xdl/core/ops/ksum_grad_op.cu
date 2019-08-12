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

#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/common_defines.h"
#include "xdl/core/lib/atomic.h"
#include "xdl/core/lib/binary_search.h"
#include "xdl/core/framework/gpu/gpu_device.h"

#include <cuda_runtime_api.h>

namespace xdl {
namespace {

template <typename T, typename I>
__global__ void KSumGradKernel(const T* pgrad, const I* pidx, const T* pval,
                               const I* pgrp, size_t grp_size, size_t eb_dim,
                               bool average, T* pout) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pgrp[grp_size - 1]) return;

  const I* p = LowerBound(pgrp, pgrp + grp_size, idx + 1);
  size_t grp_idx = p - pgrp;
  size_t grp_width = (grp_idx == 0) ? p[0]
                                    : (p[0] - p[-1]);
  if (grp_width == 0) return;

  const T* src = pgrad + grp_idx * eb_dim;
  T* dst = pout + pidx[idx] * eb_dim;
  for (size_t k = 0; k < eb_dim; ++k) {
    T val = (pval != nullptr) ? pval[idx] * src[k] : src[k];
    if (average) val /= grp_width;
    common::gpu_atomic_add<T>(val, dst + k);
  }
}

}  // namespace

template <typename T, typename I>
class KSumGradGpuOp : public GpuOpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override;
  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) override;
 private:
  bool average_;
};

template <typename T, typename I>
Status KSumGradGpuOp<T, I>::Init(OpKernelConstruction* ctx) {
  XDL_CHECK_STATUS(ctx->GetAttr("average", &average_));
  return Status::Ok();
}

template <typename T, typename I>
Status KSumGradGpuOp<T, I>::LaunchKernel(OpKernelContext* ctx, CudaStream* stream) {
  Tensor embed, index, value, segment, group, grad, out_grad;
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
  XDL_CHECK_STATUS(ctx->GetInput(7, &grad));
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

  //std::vector<int64_t> dims;
  //dims.resize(embed.Shape().NumElements());
  int64_t* pe = embed.Raw<int64_t>();
  //CUDA_CHECK(cudaMemcpy((void*)dims.data(), (void*)pe, 8 * embed.Shape().NumElements(), cudaMemcpyDeviceToHost));  
  std::vector<size_t> sdims;
  for (size_t i = 0; i < embed.Shape().NumElements(); ++i) {
    sdims.push_back(pe[i]);
  }
  //for (auto item: dims) {
  //  sdims.push_back(item);
  //}
  TensorShape embed_shape(sdims);
  size_t eb_dim = embed_shape[1];
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

  XDL_CHECK_STATUS(ctx->AllocateOutput(0, embed_shape, &out_grad));
  T* pout = out_grad.Raw<T>();
  size_t bytes = sizeof(T) * embed_shape.NumElements();
  CUDA_CHECK(cudaMemsetAsync(pout, 0, bytes, stream->GetInternal()));
  if (id_size == 0) return Status::Ok();

  size_t blocks = CUDA_GET_BLOCKS(id_size);
  KSumGradKernel<T, I><<<
      blocks,
      CUDA_GET_THREADS(id_size, blocks),
      0,
      stream->GetInternal()>>>(pgrad, pidx, pval, pgrp, grp_size,
                               eb_dim, average_, pout);

  return Status::Ok();
}

#define REGISTER_GPU_KERNEL(T, I)                    \
  XDL_REGISTER_KERNEL(KSumGrad, KSumGradGpuOp<T, I>) \
  .Device("GPU")                                     \
  .AttrDataType<T>("dtype")                          \
  .AttrDataType<I>("itype")

REGISTER_GPU_KERNEL(float, int32_t);
REGISTER_GPU_KERNEL(float, int64_t);
REGISTER_GPU_KERNEL(double, int32_t);
REGISTER_GPU_KERNEL(double, int64_t);
 
#undef REGISTER_GPU_KERNEL
}  // namespace xdl

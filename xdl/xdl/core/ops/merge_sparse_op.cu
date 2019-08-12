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

#include "xdl/core/ops/merge_sparse_op.h"

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/common_defines.h"
#include "xdl/core/lib/atomic.h"
#include "xdl/core/framework/gpu/gpu_device.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace {

template <typename I>
__global__ void MergeGroupKernel(I** seg_list,
                                 size_t seg_size,
                                 size_t grp_size,
                                 size_t num,
                                 I* out_seg,
                                 I* out_grp) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num) return;
  int grp_idx = idx / seg_size;
  int seg_idx = idx % seg_size;
  int grp_off = seg_idx * grp_size + grp_idx;

  const I* pseg = seg_list[grp_idx];
  I id_cnt = (seg_idx == 0) ? pseg[0] 
                            : (pseg[seg_idx] - pseg[seg_idx - 1]);
  common::gpu_atomic_add<I>(id_cnt, out_seg + seg_idx);
  common::gpu_atomic_add<I>(id_cnt, out_grp + grp_off);
}

template <typename I>
__global__ void ReduceKernel(I* group, size_t size) {
  int idx = threadIdx.x + blockIdx.x  * blockDim.x;
  if (idx >= size) return;
  for (size_t i = 1; i < size; ++i) {
    group[i] += group[i-1];
  }
}

template <typename T, typename V, typename I>
__global__ void MergeSparseKernel(T** id_list,
                                  V** val_list,
                                  I** seg_list,
                                  I* pgrp,
                                  size_t seg_size,
                                  size_t grp_size,
                                  size_t num,
                                  size_t id_dim,
                                  T* out_id,
                                  V* out_val) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num) return;
  int grp_idx = idx / seg_size;
  int seg_idx = idx % seg_size;
  int grp_off = seg_idx * grp_size + grp_idx;

  const T* pid = id_list[grp_idx];
  const V* pval = val_list[grp_idx];
  const I* pseg = seg_list[grp_idx];

  I src_off = seg_idx == 0 ? 0 : pseg[seg_idx - 1];
  I dst_off = grp_off == 0 ? 0 : pgrp[grp_off - 1];
  I id_cnt = grp_off == 0 ? pgrp[0] :
                            pgrp[grp_off] - pgrp[grp_off - 1];

  for (I i = 0; i < id_cnt; ++i) {
    for (size_t j = 0; j < id_dim; ++j) {
      out_id[(dst_off + i) * id_dim + j] = pid[(src_off + i) * id_dim +j];
    }
    if (pval) out_val[dst_off + i] = pval[src_off + i];
  }
}

}  // namespace

template <typename T, typename V, typename I>
class MergeSparseGpuOp : public GpuOpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }
  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) override;
};

template <typename T, typename V, typename I>
Status MergeSparseGpuOp<T, V, I>::LaunchKernel(OpKernelContext* ctx,
                                               CudaStream* stream) {
  std::vector<Tensor> id_list, value_list, segment_list;
  Tensor ids, values, segments, groups;
  XDL_CHECK_STATUS(ctx->GetInputList("id_list", &id_list));
  XDL_CHECK_STATUS(ctx->GetInputList("value_list", &value_list));
  XDL_CHECK_STATUS(ctx->GetInputList("segment_list", &segment_list));
  XDL_CHECK_COND(id_list.size() == value_list.size() &&
                 id_list.size() == segment_list.size(),
                 Status::ArgumentError("input list size not equal"));
  for (size_t i = 1; i < segment_list.size(); ++i) {
    XDL_CHECK_COND(segment_list[i].Shape().NumElements() ==
                   segment_list[i-1].Shape().NumElements(),
                   Status::ArgumentError("merged sample size must be equal"));
  }

  size_t id_size = 0, value_size = 0, seg_size = segment_list[0].Shape()[0];
  for (size_t i = 0; i < id_list.size(); ++i) {
    XDL_CHECK_COND(id_list[i].Shape()[0] == value_list[i].Shape()[0] ||
                   value_list[i].Shape().NumElements() == 0,
                   Status::ArgumentError("id and value size must be equal"));
    id_size += id_list[i].Shape()[0];
    value_size += value_list[i].Shape()[0];
  }
  std::vector<V*> value_ptr(value_list.size());
  for (size_t i = 0; i < value_list.size(); ++i) {
    if (value_list[i].Shape().NumElements() == 0) {
      XDL_CHECK(value_size == 0) << "must be all empty values";
      value_ptr[i] = nullptr;
    } else {
      value_ptr[i] = value_list[i].Raw<V>();
    }
  }
  TensorShape id_shape(id_list[0].Shape().Dims());
  id_shape.Set(0, id_size);
  TensorShape value_shape({value_size});
  TensorShape segment_shape({seg_size});
  size_t group_size = id_list.size();
  TensorShape group_shape({group_size * seg_size});

  size_t id_num = id_list[0].Shape().NumElements() / id_list[0].Shape()[0];

  XDL_CHECK_STATUS(ctx->AllocateOutput(0, id_shape, &ids));
  XDL_CHECK_STATUS(ctx->AllocateOutput(1, value_shape, &values));
  XDL_CHECK_STATUS(ctx->AllocateOutput(2, segment_shape, &segments));
  XDL_CHECK_STATUS(ctx->AllocateOutput(3, group_shape, &groups));

  T* pid = ids.Raw<T>();
  V* pvalue = values.Raw<V>();
  I* pseg = segments.Raw<I>();
  I* pgrp = groups.Raw<I>();
  
  GpuDevice* dev = dynamic_cast<GpuDevice*>(ctx->GetDevice());
  XDL_CHECK(dev != nullptr) << "gpu device is nullptr";
  cudaStream_t st = stream->GetInternal();

  T** pid_list = reinterpret_cast<T**>(dev->Allocate(sizeof(T*) * group_size));
  V** pval_list = reinterpret_cast<V**>(dev->Allocate(sizeof(V*) * group_size));
  I** pseg_list = reinterpret_cast<I**>(dev->Allocate(sizeof(I*) * group_size));
  for (size_t i = 0; i < group_size; ++i) {
    T* id = id_list[i].Raw<T>();
    CUDA_CHECK(cudaMemcpyAsync(&pid_list[i], &id, sizeof(T*), 
                               cudaMemcpyHostToDevice, st));
    V* val = value_ptr[i];
    CUDA_CHECK(cudaMemcpyAsync(&pval_list[i], &val, sizeof(V*),
                               cudaMemcpyHostToDevice, st));
    I* seg = segment_list[i].Raw<I>();
    CUDA_CHECK(cudaMemcpyAsync(&pseg_list[i], &seg, sizeof(I*),
                               cudaMemcpyHostToDevice, st));
  }

  size_t num = seg_size * group_size;
  size_t blocks = CUDA_GET_BLOCKS(num);
  MergeGroupKernel<I><<<
      blocks,
      CUDA_GET_THREADS(num, blocks),
      0,
      st>>>(pseg_list, seg_size, group_size, num, pseg, pgrp);
  ReduceKernel<I><<<1, 1, 0, st>>>(pseg, seg_size);
  ReduceKernel<I><<<1, 1, 0, st>>>(pgrp, num);
  MergeSparseKernel<T, V, I><<<
      blocks,
      CUDA_GET_THREADS(num, blocks),
      0,
      st>>>(pid_list, pval_list, pseg_list, pgrp,
            seg_size, group_size, num, id_num, pid, pvalue);

  return Status::Ok();
}

#define REGISTER_GPU_KERNEL(T, V, I)                            \
  XDL_REGISTER_KERNEL(MergeSparseOp, MergeSparseGpuOp<T, V, I>) \
  .Device("GPU")                                                \
  .AttrDataType<T>("dtype")                                     \
  .AttrDataType<V>("vtype")                                     \
  .AttrDataType<I>("itype");

REGISTER_GPU_KERNEL(int32_t, float, int32_t);
REGISTER_GPU_KERNEL(int64_t, float, int32_t);

#undef REGISTER_GPU_KERNEL

}  // namespace xdl

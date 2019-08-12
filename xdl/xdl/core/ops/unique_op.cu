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
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/atomic.h"
#include "xdl/core/lib/binary_search.h"

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/system/cuda/execution_policy.h>
#include "xdl/core/framework/gpu/gpu_device.h"
#include "xdl/core/lib/common_defines.h"

#include <chrono>

namespace xdl {
namespace functor {

template <typename T, typename I>
struct UniqueFunctor<GpuDevice, T, I> {
  void operator()(GpuDevice* d, const Tensor& in, const Tensor& segment, Tensor* out, Tensor* out_index, Tensor* sample_index, Tensor* sample_segment);
};

template <typename T>
struct Less {
  __host__ __device__ bool operator()(const thrust::pair<T, T>& l,
                                      const thrust::pair<T, T>& r) {
    return l.first < r.first || (l.first == r.first && l.second < r.second);
  }
};

template <typename T>
struct Equal {
  __host__ __device__ bool operator()(const thrust::pair<T, T>& l,
                                      const thrust::pair<T, T>& r) {
    return l.first == r.first && l.second == r.second;
  }
};

template <typename T, typename I>
__global__ void FindIndex(const T* src, size_t sz, const T* uniqs,
                          size_t uniq_sz, I* out_index, I* sample_segment) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= sz) return;
  out_index[idx] = static_cast<I>(BinarySearch(uniqs, uniq_sz, src[idx]));
  common::gpu_atomic_add<I>(1, sample_segment + out_index[idx]);
}

template <typename T, typename I>
__global__ void FindPairIndex(const T* src, size_t sz, const T* uniqs,
                              size_t uniq_sz, I* out_index, I* sample_segment) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= sz) return;
  out_index[idx] = static_cast<I>(BinarySearch2(uniqs, uniq_sz,
                                                src[2*idx], src[2*idx+1]));
  common::gpu_atomic_add<I>(1, sample_segment + out_index[idx]);
}

template <typename I>
__global__ void FindSampleIndex(const I* segment, const I* out_index, size_t sz, size_t uniq_size, size_t segment_size,
                                I* cur, I* sample_index, I* sample_segment) {
  cur[0] = 0;
  for (size_t i = 1; i < uniq_size; ++i) {
    cur[i] = sample_segment[i-1];
    sample_segment[i] += sample_segment[i-1];
  }
  I segment_idx = 0;
  for (I i = 0; i < sz; ++i) {
    while (i == *segment) {
      if (++segment_idx > segment_size) return;
      ++segment;
    }
    sample_index[cur[out_index[i]]] = segment_idx;
    cur[out_index[i]]++;
  }
}

template <typename T, typename I>
void UniqueFunctor<GpuDevice, T, I>::operator()(GpuDevice* d,
                                                const Tensor& in,
                                                const Tensor& segment,
                                                Tensor* out,
                                                Tensor* out_index,
                                                Tensor* sample_index,
                                                Tensor* sample_segment) {
  cudaStream_t stream = d->Stream()->GetInternal();
  //CUDA_CHECK(cudaStreamSynchronize(stream));
  //auto t0 = std::chrono::high_resolution_clock::now();
  Tensor temp(d, in.Shape(), in.Type());
  *out_index = Tensor(d, TensorShape({in.Shape()[0]}), DataTypeToEnum<I>::v());
  *sample_index = Tensor(d, TensorShape({in.Shape()[0]}), DataTypeToEnum<I>::v());
  T* ptr_in = in.Raw<T>();
  T* ptr_temp = temp.Raw<T>();
  CUDA_CHECK(cudaMemcpyAsync(ptr_temp,
                             ptr_in,
                             in.Shape().NumElements() * sizeof(T),
                             cudaMemcpyDeviceToDevice));
  size_t id_num = in.Shape()[0];
  size_t id_dim = in.Shape().Size() == 1 ? 1 : in.Shape()[1];
  size_t segment_size = segment.Shape()[0];
  if (id_dim == 1) {
    thrust::device_ptr<T> dptr_temp(ptr_temp), dptr_end;
    thrust::sort(thrust::cuda::par.on(stream),
                 dptr_temp, dptr_temp + id_num);
    dptr_end = thrust::unique(thrust::cuda::par.on(stream),
                              dptr_temp, dptr_temp + id_num);
    size_t uniq_size = dptr_end - dptr_temp;
    TensorShape out_shape({uniq_size});
    *out = Tensor(d, out_shape, in.Type());
    TensorShape sseg_shape({uniq_size});
    *sample_segment = Tensor(d, sseg_shape, out_index->Type());
    Tensor cur(d, sseg_shape, out_index->Type());
    CUDA_CHECK(cudaMemsetAsync(sample_segment->Raw<I>(), 0, sizeof(I) * uniq_size, stream));
    size_t blocks = CUDA_GET_BLOCKS(id_num);
    FindIndex<T, I><<<
        blocks,
        CUDA_GET_THREADS(id_num, blocks),
        0,
        stream>>>(ptr_in, id_num, ptr_temp, uniq_size, out_index->Raw<I>(), sample_segment->Raw<I>());
    FindSampleIndex<I><<<
        1,
        1,
        0,
        stream>>>(segment.Raw<I>(), out_index->Raw<I>(), id_num, uniq_size, segment_size,
                  cur.Raw<I>(), sample_index->Raw<I>(), sample_segment->Raw<I>());
    CUDA_CHECK(cudaMemcpyAsync(out->Raw<T>(),
                               ptr_temp,
                               out_shape.NumElements() * sizeof(T),
                               cudaMemcpyDeviceToDevice));
  } else if (id_dim == 2) {
    thrust::pair<T, T>* ptr_pair = reinterpret_cast<thrust::pair<T, T>*>(ptr_temp);
    thrust::device_ptr<thrust::pair<T, T>> dptr_temp(ptr_pair), dptr_end;
    thrust::sort(thrust::cuda::par.on(stream), dptr_temp, dptr_temp + id_num,
                 Less<T>());
    dptr_end = thrust::unique(thrust::cuda::par.on(stream), dptr_temp, dptr_temp + id_num,
                   Equal<T>());
    size_t uniq_size = dptr_end - dptr_temp;
    TensorShape out_shape({uniq_size, 2});
    *out = Tensor(d, out_shape, in.Type());
    TensorShape sseg_shape({uniq_size});
    *sample_segment = Tensor(d, sseg_shape, out_index->Type());
    Tensor cur(d, sseg_shape, out_index->Type());
    CUDA_CHECK(cudaMemsetAsync(sample_segment->Raw<I>(), 0, sizeof(I) * uniq_size, stream));
    size_t blocks = CUDA_GET_BLOCKS(id_num);
    FindPairIndex<T, I><<<
        blocks,
        CUDA_GET_THREADS(id_num, blocks),
        0,
        stream>>>(ptr_in, id_num, ptr_temp, uniq_size, out_index->Raw<I>(), sample_segment->Raw<I>());
    FindSampleIndex<I><<<
        1,
        1,
        0,
        stream>>>(segment.Raw<I>(), out_index->Raw<I>(), id_num, uniq_size, segment_size,
                  cur.Raw<I>(), sample_index->Raw<I>(), sample_segment->Raw<I>());
    CUDA_CHECK(cudaMemcpyAsync(out->Raw<T>(),
                               ptr_temp,
                               out_shape.NumElements() * sizeof(T),
                               cudaMemcpyDeviceToDevice));
  }

  //CUDA_CHECK(cudaStreamSynchronize(stream));
  //auto t1 = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double, std::milli> diff = t1 - t0;
  //LOG(INFO) << "unique op time:" << diff.count() << "ms, size=" << id_num;
}

template struct UniqueFunctor<GpuDevice, int64_t, int64_t>;
template struct UniqueFunctor<GpuDevice, int32_t, int32_t>;
template struct UniqueFunctor<GpuDevice, int64_t, int32_t>;
template struct UniqueFunctor<GpuDevice, int32_t, int64_t>;

}  // namespace functor

template <typename T, typename I>
class UniqueGpuOp : public GpuOpKernel {
 public:
  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) override;
};

template <typename T, typename I>
Status UniqueGpuOp<T, I>::LaunchKernel(OpKernelContext* ctx, CudaStream* stream) {
  Tensor input, segment, output, out_index, sample_index, sample_segment;
  XDL_CHECK_STATUS(ctx->GetInput(0, &input));
  XDL_CHECK_STATUS(ctx->GetInput(1, &segment));
  XDL_CHECK_COND(2 >= input.Shape().Size(),
                 Status::ArgumentError("input dim can't be greater than 2"));
  TensorShape index_shape({input.Shape()[0]});

  GpuDevice* device = dynamic_cast<GpuDevice*>(ctx->GetDevice());
  auto fn = functor::UniqueFunctor<GpuDevice, T, I>();
  fn(device, input, segment, &output, &out_index, &sample_index, &sample_segment);

  ctx->SetOutput(0, output);
  ctx->SetOutput(1, out_index);
  ctx->SetOutput(2, sample_index);
  ctx->SetOutput(3, sample_segment);
  return Status::Ok();
}

#define REGISTER_GPU_KERNEL(T, I)                \
  XDL_REGISTER_KERNEL(Unique, UniqueGpuOp<T, I>) \
    .Device("GPU")                               \
    .AttrDataType<T>("dtype")                    \
    .AttrDataType<I>("itype")

REGISTER_GPU_KERNEL(int64_t, int64_t);
REGISTER_GPU_KERNEL(int32_t, int32_t);
REGISTER_GPU_KERNEL(int64_t, int32_t);
REGISTER_GPU_KERNEL(int32_t, int64_t);

#undef REGISTER_GPU_KERNEL

}  // namespace xdl

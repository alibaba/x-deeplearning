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

#include "xdl/core/ops/unique_op.h"
#include "xdl/core/framework/op_registry.h"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include "xdl/core/framework/gpu/gpu_device.h"
#include "xdl/core/lib/common_defines.h"

namespace xdl {
namespace functor {

template <typename T, typename I, int dim>
struct UniqueLess;

template <typename T, typename I>
struct UniqueLess<T, I, 1> {
  UniqueLess(const T* data) : data_(data) {}
  __host__ __device__ bool operator()(const I& lhs, const I& rhs) {
    return data_[lhs] < data_[rhs];
  }
  const T* data_;
};

template <typename T, typename I>
struct UniqueLess<T, I, 2> {
  UniqueLess(const T* data) : data_(data) {}
  __host__ __device__ bool operator()(const I& lhs, const I& rhs) {
    return (data_[lhs * 2] < data_[rhs * 2]) || 
           (data_[lhs * 2] == data_[rhs * 2] && 
            data_[lhs * 2 + 1] < data_[rhs * 2 + 1]);
  }
  const T* data_;
};

template <typename T, int dim>
struct UniqueEqual;

template <typename T>
struct UniqueEqual<T, 1> {
  __host__ __device__ bool operator()(const T* raw, const T* uniq) {
    return raw[0] == uniq[0];
  }
};

template <typename T> 
struct UniqueEqual<T, 2> {
  __host__ __device__ bool operator()(const T* raw, const T* uniq) {
    return raw[0] == uniq[0] && raw[1] == uniq[1];
  }
};

template <typename T, typename I>
void UniqueFunctor<GpuDevice, T, I>::operator()(GpuDevice* d,
                                                const Tensor& in,
                                                Tensor* out,
                                                Tensor& out_index) {
  size_t id_num = in.Shape()[0];
  size_t id_dim = in.Shape().Size() == 1 ? 1 : in.Shape()[1];
  size_t total_num = in.Shape().NumElements();
  T* tin = in.Raw<T>();
  std::vector<I> index(id_num);
  for (size_t i = 0; i < id_num; ++i) {
    index[i] = static_cast<I>(i);
  }
  size_t bytes = sizeof(I) * id_num;
  cudaStream_t stream = d->Stream()->GetInternal();
  I* d_index = reinterpret_cast<I*>(d->Allocate(bytes));
  CUDA_CHECK(cudaMemcpyAsync(d_index,
                             index.data(),
                             bytes,
                             cudaMemcpyHostToDevice,
                             stream));
  thrust::device_ptr<I> dptr_i(d_index);
  if (id_dim == 1) {
    thrust::sort(thrust::cuda::par.on(stream),
                 dptr_i,
                 dptr_i + id_num,
                 UniqueLess<T, I, 1>(tin));
  } else if (id_dim == 2) {
    thrust::sort(thrust::cuda::par.on(stream),
                 dptr_i,
                 dptr_i + id_num,
                 UniqueLess<T, I, 2>(tin));
  }

  std::vector<T> values(total_num), output(total_num);
  std::vector<I> output_index(id_num);
  CUDA_CHECK(cudaMemcpyAsync(index.data(),
                             d_index,
                             bytes,
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(values.data(),
                             tin,
                             sizeof(T) * total_num,
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  size_t k = 0;
  if (id_dim == 1) {
    auto equal_fn = UniqueEqual<T, 1>();
    for (size_t i = 0; i < id_num; ++i) {
      if (k == 0 || !equal_fn(values.data() + index[i],
                             output.data() + k - 1)) {
        output[k++] = values[index[i]];
      }
      output_index[index[i]] = k - 1;
    }
  } else {
    auto equal_fn = UniqueEqual<T, 2>(); 
    for (size_t i = 0; i < id_num; ++i) {
      if (k == 0 || !equal_fn(values.data() + index[i] * 2,
                             output.data() + (k - 1) * 2)) {
        output[k * 2] = values[index[i] * 2];
        output[k * 2 + 1] = values[index[i] * 2 + 1];
        ++k;
      }
      output_index[index[i]] = k - 1;
    }
  }
  std::vector<size_t> shape({k});
  if (id_dim > 1) shape.push_back(id_dim);
  TensorShape out_shape(shape);
  *out = Tensor(d, out_shape, DataTypeToEnum<T>::v());
  CUDA_CHECK(cudaMemcpyAsync(out->Raw<T>(),
                             output.data(),
                             sizeof(T) * out_shape.NumElements(),
                             cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(out_index.Raw<I>(),
                             output_index.data(),
                             bytes,
                             cudaMemcpyHostToDevice,
                             stream));
  d->Deallocate(d_index);
  CUDA_CHECK(cudaStreamSynchronize(stream));
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
  Tensor input, output, out_index;
  XDL_CHECK_STATUS(ctx->GetInput(0, &input));
  XDL_CHECK_COND(2 >= input.Shape().Size(),
                 Status::ArgumentError("input dim can't be greater than 2"));
  TensorShape index_shape({input.Shape()[0]});
  XDL_CHECK_STATUS(ctx->AllocateOutput(1, index_shape, &out_index));

  GpuDevice* device = dynamic_cast<GpuDevice*>(ctx->GetDevice());
  auto fn = functor::UniqueFunctor<GpuDevice, T, I>();
  fn(device, input, &output, out_index);

  ctx->SetOutput(0, output);
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

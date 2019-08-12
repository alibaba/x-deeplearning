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

#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/common_defines.h"
#include "xdl/core/lib/atomic.h"
#include "xdl/core/framework/gpu/gpu_device.h"

namespace xdl {
namespace {

template <typename T, typename I>
__global__ void TakeGradOpKernel(const T* pin,
                                 const I* pind,
                                 size_t col,
                                 size_t num,
                                 T* pout) {
  const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num) return;
  const size_t i = k / col, j = k % col;
  common::gpu_atomic_add<T>(pin[k], pout + pind[i] * col + j);
}

}  // namespace 

template <typename T, typename I>
class TakeGradGpuOp : public GpuOpKernel {
 public:
  Status LaunchKernel(OpKernelContext* ctx, CudaStream* stream) override;
};

template <typename T, typename I>
Status TakeGradGpuOp<T, I>::LaunchKernel(OpKernelContext* ctx,
                                         CudaStream* stream) {
  Tensor grad, indicator, feature, output;
  XDL_CHECK_STATUS(ctx->GetInput(0, &grad));
  XDL_CHECK_STATUS(ctx->GetInput(1, &indicator));
  XDL_CHECK_STATUS(ctx->GetInput(2, &feature));
  XDL_CHECK_COND(1 == indicator.Shape().Size(),
                 Status::ArgumentError("indicator must be rank 1 tensor"));
  XDL_CHECK_COND(grad.Shape()[0] == indicator.Shape().NumElements(),
                 Status::ArgumentError("input and indicator size not match"));

  auto grad_dims = grad.Shape().Dims();
  size_t row = grad_dims[0];
  size_t col = grad.Shape().NumElements() / row;
  size_t num = grad.Shape().NumElements();
  T* pin = grad.Raw<T>();
  I* pind = indicator.Raw<I>();
  std::vector<size_t> dims(grad_dims.begin(), grad_dims.end());
  int64_t* pf = feature.Raw<int64_t>();
  int64_t dim = *pf;
  //CUDA_CHECK(cudaMemcpy((void*)&dim, (void*)pf, 8, cudaMemcpyDeviceToHost));
  dims[0] = dim;
  TensorShape out_shape(dims);
  XDL_CHECK_STATUS(ctx->AllocateOutput(0, out_shape, &output));
  T* pout = output.Raw<T>();

  cudaStream_t st = stream->GetInternal();
  CUDA_CHECK(cudaMemsetAsync(pout, 0, sizeof(T) * out_shape.NumElements(), st));
  if (num == 0) {
    return Status::Ok();
  }
  size_t blocks = CUDA_GET_BLOCKS(num);
  TakeGradOpKernel<T, I><<<
      blocks,
      CUDA_GET_THREADS(num, blocks),
      0,
      st>>>(pin, pind, col, num, pout);

  return Status::Ok();
}

#define REGISTER_GPU_KERNEL(T, I)                    \
  XDL_REGISTER_KERNEL(TakeGrad, TakeGradGpuOp<T, I>) \
  .Device("GPU")                                     \
  .AttrDataType<T>("dtype")                          \
  .AttrDataType<I>("itype")

REGISTER_GPU_KERNEL(float, int32_t);
REGISTER_GPU_KERNEL(float, int64_t);
REGISTER_GPU_KERNEL(double, int32_t);
REGISTER_GPU_KERNEL(double, int64_t);

#undef REGISTER_GPU_KERNEL

}  // namespace xdl

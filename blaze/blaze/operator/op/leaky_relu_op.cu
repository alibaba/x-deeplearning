/*
 * \file leaky_relu_op.cu
 * \desc The leaky relu operator.
 */
#include "blaze/operator/op/leaky_relu_op.h"
#include "blaze/math/activation.h"

namespace blaze {

template <typename DType>
__global__ void LeakyReluKernel(LeakyReluParam<DType> params) {
  Activation<kRelu> activation;
  CUDA_KERNEL_LOOP(index, params.size) {
    activation(params.alpha, params.x + index, params.y + index);
  }
}

template <>
bool LeakyReluOp<CUDAContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH_ON_CUDA(X->data_type(), DType, {
  
  // Reshape
  Y->Reshape(X->shape());

  // Lauch the kernel
  dim3 grid, block;
  block.x = GetThreadsNum(X->size());
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(X->size(), block.x));

  cudaStream_t stream = this->context_.cuda_stream();
  LeakyReluParam<DType> params(X->as<DType>(), X->size(), Y->as<DType>(), alpha_);
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(LeakyReluKernel<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });
  return true;
}

REGISTER_CUDA_OPERATOR(LeakyRelu, LeakyReluOp<CUDAContext>);

}  // namespace blaze

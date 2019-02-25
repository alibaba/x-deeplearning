/*
 * \file sigmoid_op.cu
 * \desc The sigmoid operator
 */
#include "blaze/operator/op/sigmoid_op.h"

#include "blaze/math/activation.h"

namespace blaze {

template <typename DType>
__global__ void SigmoidKernel(SigmoidParam<DType> params) {
  Activation<kSigmoid> activation;
  CUDA_KERNEL_LOOP(index, params.size) {
    activation(params.x + index, params.y + index);
  }
}

template <>
bool SigmoidOp<CUDAContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH_ON_CUDA(X->data_type(), DType, {
  
  // Reshape
  Y->Reshape(X->shape());

  // lauch the kernel
  dim3 grid, block;
  block.x = GetThreadsNum(X->size());
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(X->size(), block.x));

  cudaStream_t stream = this->context_.cuda_stream();
  SigmoidParam<DType> params(X->as<DType>(), X->size(), Y->as<DType>());
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&SigmoidKernel<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });  // TYPE_SWITCH(Y->data_type(), DType, {

  return true;
}

REGISTER_CUDA_OPERATOR(Sigmoid, SigmoidOp<CUDAContext>);

}  // namespace blaze

/*
 * \file not_op.cu
 * \desc The not operator
 */
#include "blaze/operator/op/not_op.h"

namespace blaze {

template <typename DType>
__global__ void NotKernel(NotParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.size) {
    params.y[index] = params.x[index] == 0 ? 1 : 0;
  }
}

template <>
bool NotOp<CUDAContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH_ON_CUDA(X->data_type(), DType, {
  
  // Reshape
  Y->Reshape(X->shape());

  // lauch the kernel
  dim3 grid, block;
  block.x = GetThreadsNum(X->size());
  grid.x = CUDA_GET_BLOCKS(X->size(), block.x);

  cudaStream_t stream = this->context_.cuda_stream();
  NotParam<DType> params(X->as<DType>(), X->size(), Y->as<DType>());
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&NotKernel<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });  // TYPE_SWITCH(Y->data_type(), DType, {

  return true;
}

REGISTER_CUDA_OPERATOR(Not, NotOp<CUDAContext>);

}  // namespace blaze

/*
 * \file tanh_op.cc 
 * \brief The tanh operation
 */
#include "blaze/operator/op/tanh_op.h"

namespace blaze {

template <typename DType>
struct TanhParam {
  DType* x;
  size_t size;
  DType* y;

  TanhParam(DType* x, size_t size, DType* y) :
      x(x), size(size), y(y) { }
};

template <typename DType>
__global__ void TanhKernel(TanhParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.size) {
    params.y[index] = tanh(params.x[index]);
  }
}

template <>
bool TanhOp<CUDAContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH_ON_CUDA(X->data_type(), DType, {
  // Reshape
  Y->Reshape(X->shape());

  // Launch the kernel
  dim3 grid, block;
  block.x = GetThreadsNum(X->size());
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(X->size(), block.x));

  cudaStream_t stream = this->context_.cuda_stream();
  TanhParam<DType> params(X->as<DType>(), X->size(), Y->as<DType>());
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&TanhKernel<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });

  return true;
}

REGISTER_CUDA_OPERATOR(Tanh, TanhOp<CUDAContext>);

}  // namespace blaze


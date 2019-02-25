/*
 * \file ulf_fuse_op.cu 
 * \brief The ulf fuse operation
 */
#include "blaze/operator/op/fuse_op.h"

namespace blaze {

template <typename DType>
__global__ void FuseKernel(FuseParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.y_m * params.y_n) {
    size_t row = index / params.y_n;
    size_t col = index % params.y_n;
    params.y[index] = col < params.x1_n ?
        params.x1[params.x1_m == 1 ? col : row * params.x1_n + col] :
        params.x2[params.x1_m != 1 ? col - params.x1_n : row * params.x2_n + col - params.x1_n];
  }
}

template <>
bool FuseOp<CUDAContext>::RunOnDevice() {
  CheckValid();
  Blob* x = this->Input(0);
  TYPE_SWITCH_ON_CUDA(x->data_type(), DType, {
  // setup params
  FuseParam<DType> params;
  Setup<DType>(&params);
  // launch the kernel
  dim3 grid, block;
  block.x = GetThreadsNum(params.y_m * params.y_n);
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(params.y_m * params.y_n, block.x));

  cudaStream_t stream = this->context_.cuda_stream();
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&FuseKernel<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,  // can use shared memory because the user copy many times.
                              stream));
  });
  return true;
}

REGISTER_CUDA_OPERATOR(Fuse, FuseOp<CUDAContext>);

}  // namespace blaze

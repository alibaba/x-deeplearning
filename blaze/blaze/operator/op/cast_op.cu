/*
 * \file cast_op.cu
 * \desc The cast operator
 */
#include "blaze/operator/op/cast_op.h"

namespace blaze {

template <typename SrcDType, typename DstDType>
__global__ void CastKernel(CastParam<SrcDType, DstDType> params) {
  CUDA_KERNEL_LOOP(index, params.size) {
    params.y[index] = params.x[index];
  }
}

template <>
bool CastOp<CUDAContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH_ON_CUDA(X->data_type(), SrcDType, {
  TYPE_SWITCH_ON_CUDA(Y->data_type(), DstDType, {

  // Reshape
  Y->Reshape(X->shape());

  // lauch the kernel
  dim3 grid, block;
  block.x = GetThreadsNum(X->size());
  grid.x = CUDA_GET_BLOCKS(X->size(), block.x);

  cudaStream_t stream = this->context_.cuda_stream();
  CastParam<SrcDType, DstDType> params(X->as<SrcDType>(), X->size(), Y->as<DstDType>());
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&CastKernel<SrcDType, DstDType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });  // TYPE_SWITCH(X->data_type(), SrcDType, {
  });  // TYPE_SWITCH(Y->data_type(), DstDType, {

  return true;
}

REGISTER_CUDA_OPERATOR(Cast, CastOp<CUDAContext>);

}  // namespace blaze

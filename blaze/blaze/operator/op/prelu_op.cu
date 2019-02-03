/*
 * \file prelu_op.cu
 * \desc The prelu operator.
 */
#include "blaze/operator/op/prelu_op.h"
#include "blaze/math/activation.h"

namespace blaze {

template <typename DType, typename WDType>
__global__ void RunPRelu(PReluParam<DType, WDType> params) {
  //
  // NOTE: we not fully implement ONNX standerd, just simply deems
  // The W_Tensor is the inner prelu.
  //
  // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md.
  // PRelu fully support Unidirectional Broadcasting
  Activation<kPRelu> activation;
  CUDA_KERNEL_LOOP(index, params.size) {
    size_t offset = index % params.inner_size;
    activation(params.x + index, params.w + offset, params.y + index);
  }
}

template <>
bool PReluOp<CUDAContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* W = this->Input(1);
  Blob* Y = this->Output(0);

  // Check valid
  CheckValid();

  TYPE_SWITCH_ON_CUDA(X->data_type(), DType, {
  TYPE_SWITCH_ON_CUDA(W->data_type(), WDType, {

  // reshape
  Y->Reshape(X->shape());
  
  // launch the kernel
  dim3 grid, block;
  block.x = GetThreadsNum(X->size());
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(X->size(), block.x));

  cudaStream_t stream = this->context_.cuda_stream();
  PReluParam<DType, WDType> params(X->as<DType>(), W->as<WDType>(), Y->as<DType>(),
      X->size(), W->size());
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&RunPRelu<DType, WDType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));
  
  });  // TYPE_SWITCH(W->data_type(), WDType, {
  });  // TYPE_SWITCH(X->data_type(), DType, {
  
  return true;
}

REGISTER_CUDA_OPERATOR(PRelu, PReluOp<CUDAContext>);

}  // namespace blaze

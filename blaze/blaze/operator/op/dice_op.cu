/*
 * \file dice_op.cu 
 * \brief The dice operation
 */
#include "blaze/operator/op/dice_op.h"

#include "blaze/operator/common_helper.h"

namespace blaze {

template <typename DType>
__global__ void DiceKernel(DiceParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.size) {
    size_t offset = index % params.c;
    DType x_normed = (params.x[index] - params.mean[offset]) /
        (params.nosqrt ? params.var[offset] :
         (DType)(sqrtf(params.var[offset] + kDiceEpsilon)));
    DType x_p = 1.0 / (1.0 + __expf(-x_normed));
    params.y[index] = (1.0 - x_p) * params.gamma[offset] * params.x[index] + x_p * params.x[index];
  }
}

template <>
bool DiceOp<CUDAContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  Blob* gamma = this->Input(1);
  Blob* mean = this->Input(2);
  Blob* var = this->Input(3);
  Blob* y = this->Output(0);

  // Check output valid
  CheckValid();

  TYPE_SWITCH_ON_CUDA(x->data_type(), DType, {
  
  // Reshape
  y->Reshape(x->shape());

  // Launch cuda kernel
  dim3 grid, block;
  block.x = GetThreadsNum(x->size());
  grid.x = GetBlockNum(CUDA_GET_BLOCKS(x->size(), block.x));

  cudaStream_t stream = this->context_.cuda_stream();
  DiceParam<DType> params(x->as<DType>(),
                          x->size(),
                          gamma->as<DType>(),
                          mean->as<DType>(),
                          var->as<DType>(),
                          gamma->size(),
                          y->as<DType>(),
                          nosqrt_);
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&DiceKernel<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });
  
  return true;
}

REGISTER_CUDA_OPERATOR(Dice, DiceOp<CUDAContext>);

}  // namespace blaze

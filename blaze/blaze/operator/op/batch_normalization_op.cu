/*
 * \file batch_normalization_op.cu 
 * \brief The batch normalization operation
 */
#include "blaze/operator/op/batch_normalization_op.h"

#include "blaze/operator/common_helper.h"

namespace blaze {

template <typename DType>
__global__ void BatchNormalizationKernel(BatchNormalizationParam<DType> params) {
  CUDA_KERNEL_LOOP(index, params.size) {
    size_t offset = index % params.c;
    DType x_normed = (params.x[index] - params.mean[offset]) /
        (params.nosqrt ? params.var[offset] :
         (DType)(sqrtf((float)(params.var[offset] + params.eps))));
    params.y[index] = params.gamma[offset] * x_normed + params.beta[offset];
  }
}

template <>
bool BatchNormalizationOp<CUDAContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  Blob* gamma = this->Input(1);
  Blob* beta = this->Input(2);
  Blob* mean = this->Input(3);
  Blob* var = this->Input(4);
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
  BatchNormalizationParam<DType> params(x->as<DType>(),
                                        x->size(),
                                        gamma->as<DType>(),
                                        beta->as<DType>(),
                                        mean->as<DType>(),
                                        var->as<DType>(),
                                        gamma->size(),
                                        y->as<DType>(),
                                        nosqrt_,
                                        eps_);
  void* params_dptr = reinterpret_cast<void*>(&params);
  CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<void*>(&BatchNormalizationKernel<DType>),
                              grid,
                              block,
                              reinterpret_cast<void**>(&params_dptr),
                              0,
                              stream));

  });
  
  return true;
}

REGISTER_CUDA_OPERATOR(BatchNormalization, BatchNormalizationOp<CUDAContext>);

}  // namespace blaze

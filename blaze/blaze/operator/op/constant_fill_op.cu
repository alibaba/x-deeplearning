/*
 * \file constant_fill_op.cu 
 * \brief The constant fill operation
 */
#include "blaze/operator/op/constant_fill_op.h"

#include "blaze/common/context.h"

namespace blaze {

template <>
void ConstantFillOp<CUDAContext>::CopyData(void* dst, const void* src, size_t size) {
  this->context_.SwitchToDevice(0);
  g_copy_function[kCPU][kCUDA](dst, src, 0, 0, size, (this->context_.cuda_stream()));
  this->context_.FinishDeviceComputation();
}

REGISTER_CUDA_OPERATOR(ConstantFill, ConstantFillOp<CUDAContext>);

}  // namespace blaze

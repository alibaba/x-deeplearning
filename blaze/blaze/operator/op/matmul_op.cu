/*
 * \file matmul_op.cu 
 * \brief The matmul operation
 */
#include "blaze/operator/op/matmul_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(MatMul, MatMulOp<CUDAContext>);

}  // namespace blaze

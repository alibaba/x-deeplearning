/*
 * \file fused_parallel_mul_op.cu 
 * \brief The fused parallel mul operation
 */
#include "blaze/operator/fused_op/fused_parallel_mul_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(FusedParallelMul, FusedParallelMulOp<CUDAContext>);

}  // namespace blaze


/*
 * \file fused_parallel_matmul_op.cu 
 * \brief The fused parallel matmul operation
 */
#include "blaze/operator/fused_op/fused_parallel_matmul_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(FusedParallelMatMul, FusedParallelMatMulOp<CUDAContext>);

}  // namespace blaze


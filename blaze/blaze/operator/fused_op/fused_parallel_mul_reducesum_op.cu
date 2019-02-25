/*
 * \file fused_parallel_mul_reducesum_op.cu
 * \file The fused parallel mul reducesum operation
 */
#include "blaze/operator/fused_op/fused_parallel_mul_reducesum_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(FusedParallelMulReduceSum, FusedParallelMulReduceSumOp<CUDAContext>);

}  // namespace blaze

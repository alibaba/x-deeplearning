/*
 * \file fused_parallel_gemm_op.cu 
 * \brief The fused parallel gemm operation
 */
#include "blaze/operator/fused_op/fused_parallel_gemm_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(FusedParallelGemm, FusedParallelGemmOp<CUDAContext>);

}  // namespace blaze


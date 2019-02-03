/*
 * \file pruned_gemm_op.cu
 * \desc The pruned gemm operator
 * 
 *  X1 | X2  mul  W1  + bias    = X1 x W1 + X2 x W2 + bias
 *                --
 *                W2 
 */
#include "blaze/operator/op/pruned_gemm_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(PrunedGemm, PrunedGemmOp<CUDAContext>);

}  // namespace blaze

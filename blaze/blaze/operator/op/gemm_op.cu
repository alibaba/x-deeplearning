/*
 * \file gemm_op.cu 
 * \brief The gemm operation on gpu implementation
 */
#include "blaze/operator/op/gemm_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(Gemm, GemmOp<CUDAContext>);

}  // namespace blaze

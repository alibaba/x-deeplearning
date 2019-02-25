/*
 * \file reduce_sum_op.cu 
 * \brief The reduce sum operation
 */
#include "blaze/operator/op/reduce_sum_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(ReduceSum, ReduceSumOp<CUDAContext>);

}  // namespace blaze


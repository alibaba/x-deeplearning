/*
 * \file flatten_op.cu
 * \brief The flatten operation
 */
#include "blaze/operator/op/flatten_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(Flatten, FlattenOp<CUDAContext>);

}  // namespace blaze


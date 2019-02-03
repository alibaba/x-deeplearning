/*
 * \file reshape_op.cu
 * \brief The reshape operation on gpu arch
 */
#include "blaze/operator/op/reshape_op.h"

namespace blaze {

REGISTER_CUDA_OPERATOR(Reshape, ReshapeOp<CUDAContext>);

}  // namespace blaze

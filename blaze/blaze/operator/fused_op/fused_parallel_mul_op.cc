/*
 * \file fused_parallel_mul_op.cc 
 * \brief The fused parallel mul operation
 */
#include "blaze/operator/fused_op/fused_parallel_mul_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(FusedParallelMul, FusedParallelMulOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(FusedParallelMul)
  .NumInputs(2)
  .NumOutputs(1)
  .SetDoc(R"DOC(
FusedParallelMul do mutiple Mul parallelly 
  )DOC")
  .Input(0, "X", "N-D Input tensor")
  .Output(0, "Y", "N-D output tensor");

}  // namespace blaze

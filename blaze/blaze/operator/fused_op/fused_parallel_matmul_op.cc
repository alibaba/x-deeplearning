/*
 * \file fused_parallel_matmul_op.cc 
 * \brief The fused parallel matmul operation
 */
#include "blaze/operator/fused_op/fused_parallel_matmul_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(FusedParallelMatMul, FusedParallelMatMulOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(FusedParallelMatMul)
  .NumInputs(2)
  .NumOutputs(1)
  .SetDoc(R"DOC(
FusedParallelMatMul do mutiple MatMul parallelly 
  )DOC")
  .Input(0, "X", "N-D Input tensor")
  .Output(0, "Y", "N-D output tensor");

}  // namespace blaze

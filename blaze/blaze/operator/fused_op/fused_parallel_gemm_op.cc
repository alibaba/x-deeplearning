/*
 * \file fused_parallel_gemm_op.cc 
 * \brief The fused parallel gemm operation
 */
#include "blaze/operator/fused_op/fused_parallel_gemm_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(FusedParallelGemm, FusedParallelGemmOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(FusedParallelGemm)
  .NumInputs(2, 3)
  .NumOutputs(1)
  .SetDoc(R"DOC(
FusedParallelGemm do mutiple Gemm parallelly 
  )DOC")
  .Input(0, "X", "N-D Input tensor")
  .Output(0, "Y", "N-D output tensor");

}  // namespace blaze

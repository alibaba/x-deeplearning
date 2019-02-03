/*
 * \file fused_parallel_mul_reducesum_op.cc
 * \file The fused parallel mul reducesum operation
 */
#include "blaze/operator/fused_op/fused_parallel_mul_reducesum_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(FusedParallelMulReduceSum, FusedParallelMulReduceSumOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(FusedParallelMulReduceSum)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
FusedParallelMulReduce.
    )DOC")
    .Input(0, "X", "N-D batched input tensor")
    .Output(0, "Y", "N-D batched output tensor");

}  // namespace blaze

/*
 * \file reduce_sum_op.cc 
 * \brief The reduce sum operation
 */
#include "blaze/operator/op/reduce_sum_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(ReduceSum, ReduceSumOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(ReduceSum)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
ReduceSum activation.
    )DOC")
    .Input(0, "X", "N-D input tensor")
    .Output(0, "Y", "N-D output tensor");

}  // namespace blaze


/*
 * \file reshape_op.cc 
 * \brief The reshape operation on cpu arch
 */
#include "blaze/operator/op/reshape_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(Reshape, ReshapeOp<CPUContext>);

// Input: X Shape Output: Y
OPERATOR_SCHEMA(Reshape)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Reshape the input tensor
    )DOC")
    .Input(0, "X", "1D input tensor")
    .Input(1, "Shape", "The shape to be reshaped")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze


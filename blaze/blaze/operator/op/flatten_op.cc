/*
 * \file flatten_op.cc
 * \brief The flatten operation
 */
#include "blaze/operator/op/flatten_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(Flatten, FlattenOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(Flatten)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Flatten the n-d dimention tensor into 2D Matrix.
    )DOC");

}  // namespace blaze


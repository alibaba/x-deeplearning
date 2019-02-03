/*
 * \file matmul_op.cc 
 * \brief The matmul operation
 */
#include "blaze/operator/op/matmul_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(MatMul, MatMulOp<CPUContext>);

// Input: A, B Output: C
OPERATOR_SCHEMA(MatMul)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
MatMul operator C=A*B.
    )DOC");

}  // namespace blaze

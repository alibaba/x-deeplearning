/*
 * \file pruned_gemm_op.cc
 * \desc The pruned gemm operator
 * 
 *  X1 | X2  mul  W1  + bias    = X1 x W1 + X2 x W2 + bias
 *                --
 *                W2 
 */
#include "blaze/operator/op/pruned_gemm_op.h"

namespace blaze {

REGISTER_CPU_OPERATOR(PrunedGemm, PrunedGemmOp<CPUContext>);

// Input: X1, X2, W1, W2, Bias(optional), Output: C
OPERATOR_SCHEMA(PrunedGemm)
    .NumInputs(4, 5)
    .NumOutputs(1)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
Pruned Gemm operator, If the input of Gemm has duplicated datas,
Pruned Gemm can acccelerate Gemm computation.
    )DOC");

}  // namespace blaze


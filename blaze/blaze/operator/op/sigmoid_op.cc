/*
 * \file sigmoid_op.cc
 * \desc The sigmoid operator
 */
#include "blaze/operator/op/sigmoid_op.h"

#include <omp.h>
#include <math.h>

#include "blaze/math/activation.h"
#include "blaze/math/vml.h"

namespace blaze {

template <typename DType>
void SigmoidKernel(const SigmoidParam<DType>& params) {
  VML_Exp<DType, CPUContext>(params.size, params.x, params.y, nullptr);
  for (int k = 0; k < params.size; ++k) {
    params.y[k] = params.y[k] / (1 + params.y[k]);
  }
}

template <>
bool SigmoidOp<CPUContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH(X->data_type(), DType, {
    // Reshape
    Y->Reshape(X->shape());
    // lauch cpu kernel
    SigmoidParam<DType> params(X->as<DType>(), X->size(), Y->as<DType>());
    SigmoidKernel(params);
  });
  return true;
}

REGISTER_CPU_OPERATOR(Sigmoid, SigmoidOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(Sigmoid)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeOfInput(0)
    .SetAttr<bool>(kAttrIsElementWise, true)
    .SetDoc(R"DOC(
Sigmoid activation.
    )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze


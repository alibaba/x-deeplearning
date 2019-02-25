/*
 * \file leaky_relu_op.cc
 * \desc The leaky relu operator.
 */
#include "blaze/operator/op/leaky_relu_op.h"

#include <omp.h>
#include <math.h>

#include "blaze/math/activation.h"
#include "blaze/math/vml.h"

namespace blaze {

template <typename DType>
void LeakyReluKernel(LeakyReluParam<DType>& params) {
  Activation<kRelu> activation;
  for (int k = 0; k < params.size; ++k) {
    activation(params.alpha, params.x + k, params.y + k);
  }
}

template <>
bool LeakyReluOp<CPUContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH(X->data_type(), DType, {
    // Reshape
    Y->Reshape(X->shape());
    // launch cpu kernel
    LeakyReluParam<DType> params(X->as<DType>(), X->size(), Y->as<DType>(), alpha_);
    LeakyReluKernel(params);
  });

  return true;
}

REGISTER_CPU_OPERATOR(LeakyRelu, LeakyReluOp<CPUContext>);

// Input: X, Slope  Output: Y
OPERATOR_SCHEMA(LeakyRelu)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeOfInput(0)
  .SetAttr<bool>(kAttrIsElementWise, true)
  .SetDoc(R"DOC(
LeakyRelu takes input data (Tensor) and an argument alpha, and produces one output data (Tensor) where the function f(x) = alpha * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise
  )DOC")
  .Input(0, "X", "1D input tensor")
  .Output(0, "Y", "1D output tensor");

}  // namespace blaze

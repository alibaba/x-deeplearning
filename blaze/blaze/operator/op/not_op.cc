/*
 * \file not_op.cc
 * \desc The not operator
 */
#include "blaze/operator/op/not_op.h"

namespace blaze {

template <typename DType>
void NotKernel(const NotParam<DType>& params) {
  for (size_t k = 0; k < params.size; ++k) {
    params.y[k] = params.x[k] == 0 ? 1 : 0;
  }
}

template <>
bool NotOp<CPUContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH(X->data_type(), DType, {
    // Reshape
    Y->Reshape(X->shape());
    // lauch cpu kernel
    NotParam<DType> params(X->as<DType>(), X->size(), Y->as<DType>());
    NotKernel(params);
  });
  return true;
}

REGISTER_CPU_OPERATOR(Not, NotOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(Not)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
Sigmoid activation.
    )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze


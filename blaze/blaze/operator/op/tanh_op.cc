/*
 * \file tanh_op.cc 
 * \brief The tanh operation
 */
#include "blaze/operator/op/tanh_op.h"

#include "blaze/math/vml.h"

namespace blaze {

template <>
bool TanhOp<CPUContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  Blob* y = this->Output(0);

  TYPE_SWITCH(x->data_type(), DType, {
  // Reshape
  y->Reshape(x->shape());
  // Tanh
  VML_Tanh<DType, CPUContext>(x->size(), x->as<DType>(), y->as<DType>(), &this->context_);
  });
  return true;
}

REGISTER_CPU_OPERATOR(Tanh, TanhOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(Tanh)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeOfInput(0)
  .SetAttr<bool>(kAttrIsElementWise, true)
  .SetDoc(R"DOC(
Tanh activation
  )DOC")
  .Input(0, "X", "N-D Input tensor")
  .Output(0, "Y", "N-D Output tensor");

}  // namespace blaze


/*
 * \file prelu_op.cc
 * \desc The prelu operator.
 */
#include "blaze/operator/op/prelu_op.h"
#include "blaze/math/activation.h"

namespace blaze {

template <typename DType, typename WDType>
void RunPRelu(PReluParam<DType, WDType>& params) {
  size_t inner_size = params.inner_size;
  size_t outer_size = params.size / inner_size;

  //
  // NOTE: we not fully implement ONNX standerd, just simply deems
  // The W_Tensor is the inner prelu.
  //
  // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md.
  // PRelu fully support Unidirectional Broadcasting
  //
  Activation<kPRelu> activation;
  size_t offset = 0;
  for (size_t i = 0; i < outer_size; ++i) {
    for (size_t j = 0; j < inner_size; ++j, ++offset) {
      // TODO: Add AVX SIMD support
      activation(params.x + offset, params.w + j, params.y + offset);
    }
  }
}

template <>
bool PReluOp<CPUContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* W = this->Input(1);
  Blob* Y = this->Output(0);

  // Check valid
  CheckValid();

  TYPE_SWITCH(X->data_type(), DType, {
    TYPE_SWITCH(W->data_type(), WDType, {
      // Reshape
      Y->Reshape(X->shape());
      // launch cpu kernel
      PReluParam<DType, WDType> params(X->as<DType>(), W->as<WDType>(), Y->as<DType>(),
                                       X->size(), W->size());
      RunPRelu(params);
    });
  });

  return true;
}

REGISTER_CPU_OPERATOR(PRelu, PReluOp<CPUContext>);

// Input: X, Slope  Output: Y
OPERATOR_SCHEMA(PRelu)
  .NumInputs(2)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeOfInput(0)
  .SetAttr<bool>(kAttrIsElementWise, true)
  .SetDoc(R"DOC(
PRelu takes input data (Tensor<T>) and slopes tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
  )DOC")
  .Input(0, "X", "1D input tensor")
  .Input(1, "Slope",
         "1D slope tensor, If `slope` is of size 1, the value is shared"
         "across different channels")
  .Output(0, "Y", "1D output tensor");

}  // namespace blaze

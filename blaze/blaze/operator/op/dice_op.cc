/*
 * \file dice_op.cc 
 * \brief The dice operation
 */
#include "blaze/operator/op/dice_op.h"

#include <math.h>

#include "blaze/math/vml.h"
#include "blaze/operator/common_helper.h"

namespace blaze {

template <typename DType>
void DiceKernel(DiceParam<DType>& params) {
  size_t batch_size = params.size / params.c;
  size_t offset = 0;

  for (size_t row = 0; row < batch_size; ++row) {
    for (size_t col = 0; col < params.c; ++col) {
      params.y[offset + col] = -1 * (params.x[offset + col] - params.mean[col]) /
          (params.nosqrt ? params.var[col] : sqrtf(params.var[col] + kDiceEpsilon));
    }
    VML_Exp<DType, CPUContext>(params.c, params.y + offset, params.y + offset, nullptr);
    for (size_t col = 0; col < params.c; ++col) {
      params.y[offset + col] = 1 / (1 + params.y[offset + col]);
    }
    for (size_t col = 0; col < params.c; ++col) {
      params.y[offset + col] = (1 - params.y[offset + col]) * params.gamma[col] * params.x[offset + col]
          + params.y[offset + col] * params.x[offset + col];
    }
    offset += params.c;
  }
}

template <>
bool DiceOp<CPUContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  Blob* gamma = this->Input(1);
  Blob* mean = this->Input(2);
  Blob* var = this->Input(3);
  Blob* y = this->Output(0);

  // Check input valid
  CheckValid();

  TYPE_SWITCH(x->data_type(), DType, {
  // Reshape
  y->Reshape(x->shape());
  // Launch cpu kernel
  DiceParam<DType> params(x->as<DType>(),
                          x->size(),
                          gamma->as<DType>(),
                          mean->as<DType>(),
                          var->as<DType>(),
                          gamma->size(),
                          y->as<DType>(),
                          nosqrt_);
  DiceKernel<DType>(params);
  });

  return true;
}

REGISTER_CPU_OPERATOR(Dice, DiceOp<CPUContext>);

// Input: X, gamma(scale), beta(bias), mean, var Output: Y
OPERATOR_SCHEMA(Dice)
    .NumInputs(4)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeOfInput(0)
    .CostInferenceFunction([](const OperatorDef& def,
                              const std::vector<TensorShape>& input_shape,
                              const std::vector<DataType>& input_type,
                              const std::vector<TensorShape>& output_shape,
                              const std::vector<DataType>& output_type) {
       return ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
     })
    .IdenticalShapeOfInput(0)
    .SetDoc(R"DOC(
Carries out dice as described in the paper: https://arxiv.org/pdf/1706.06978.pdf 
    )DOC")
    .Input(0, "X", "1D Input tensor")
    .Input(1, "gamma", "The gamma/scale tensor")
    .Input(2, "mean", "The mean tensor")
    .Input(3, "var", "The var tensor")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze


/*
 * \file batch_normalization_op.cc 
 * \brief The batch normalization operation
 */
#include "blaze/operator/op/batch_normalization_op.h"

#include <math.h>

#include "blaze/operator/common_helper.h"

namespace blaze {

template <typename DType>
void BatchNormalizationKernel(BatchNormalizationParam<DType>& params) {
  for (size_t i = 0; i < params.size; ++i) {
    size_t offset = i % params.c;
    DType x_normed = (params.x[i] - params.mean[offset]) /
        (params.nosqrt ? params.var[offset] : sqrtf(params.var[offset] + params.eps));
    params.y[i] = params.gamma[offset] * x_normed + params.beta[offset];
  }
}

template <>
bool BatchNormalizationOp<CPUContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  Blob* gamma = this->Input(1);
  Blob* beta = this->Input(2);
  Blob* mean = this->Input(3);
  Blob* var = this->Input(4);
  Blob* y = this->Output(0);

  // Check input valid
  CheckValid();

  TYPE_SWITCH(x->data_type(), DType, {
    // Reshape
    y->Reshape(x->shape());
    // Launch cpu kernel
    BatchNormalizationParam<DType> params(x->as<DType>(),
                                          x->size(),
                                          gamma->as<DType>(),
                                          beta->as<DType>(),
                                          mean->as<DType>(),
                                          var->as<DType>(),
                                          gamma->size(),
                                          y->as<DType>(),
                                          nosqrt_,
                                          eps_);
    BatchNormalizationKernel<DType>(params);
  });

  return true;
}

REGISTER_CPU_OPERATOR(BatchNormalization, BatchNormalizationOp<CPUContext>);

// Input: X, gamma(scale), beta(bias), mean, var Output: Y
OPERATOR_SCHEMA(BatchNormalization)
    .NumInputs(5)
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
Carries out batch normalization as described in the paper https://arxiv.org/abs/1502.03167.
    )DOC")
    .Input(0, "X", "1D Input tensor")
    .Input(1, "gamma", "The gamma/scale tensor")
    .Input(2, "beta", "The beta/bias tensor")
    .Input(3, "mean", "The mean tensor")
    .Input(4, "var", "The var tensor")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze


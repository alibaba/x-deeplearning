/*
 * \file cast_op.cc
 * \desc The not operator
 */
#include "blaze/operator/op/cast_op.h"

namespace blaze {

template <typename SrcDType, typename DstDType>
void CastKernel(const CastParam<SrcDType, DstDType>& params) {
  for (size_t k = 0; k < params.size; ++k) {
    params.y[k] = params.x[k];
  }
}

template <>
bool CastOp<CPUContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH(X->data_type(), SrcDType, {
  TYPE_SWITCH(Y->data_type(), DstDType, {
  
  // Reshape
  Y->Reshape(X->shape());
  // lauch cpu kernel
  CastParam<SrcDType, DstDType> params(X->as<SrcDType>(), X->size(), Y->as<DstDType>());
  CastKernel(params);
              
  });
  });
  return true;
}

REGISTER_CPU_OPERATOR(Cast, CastOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(Cast)
    .NumInputs(1)
    .NumOutputs(1)
    .TypeInferenceFunction([](const OperatorDef& def, const std::vector<DataType>& input_type) {
      ArgumentHelper argument_helper(def);
      std::vector<DataType> ret;
      ret.push_back(static_cast<DataType>(argument_helper.GetSingleArgument<int>("to", kFloat)));
      return ret;
    })
    .SetDoc(R"DOC(
Cast the input to another type Tensor.
    )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze


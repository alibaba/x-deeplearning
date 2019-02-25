/*
 * \file concat_op.h 
 * \brief The concat operation implementation
 */
#include "blaze/operator/op/concat_op.h"

namespace blaze {

template <typename DType>
void RunConcat(ConcatParam<DType>& params) {
  size_t y_offset = 0;
  for (size_t i = 0; i < params.outer_size; ++i) {
    for (size_t k = 0; k < params.input_size; ++k) {
      memcpy(params.y + y_offset, params.concat_item[k].x, params.concat_item[k].axis_size * sizeof(DType));
      params.concat_item[k].x += params.concat_item[k].axis_size;
      y_offset += params.concat_item[k].axis_size;
    }
  }
}

template <>
bool ConcatOp<CPUContext>::RunOnDevice() {
  // Check the validity of Concat
  CheckValid();

  Blob* x0 = this->Input(0);
  TYPE_SWITCH(x0->data_type(), DType, {

  ConcatParam<DType> params;
  // Prepare params and reshape
  Setup<DType>(&params);

  // Start to execute concat kernel
  RunConcat(params);

  });
  return true;
}

REGISTER_CPU_OPERATOR(Concat, ConcatOp<CPUContext>);

// Input: X1, X2, ... Output: Y
// Argument: axis
OPERATOR_SCHEMA(Concat)
  .NumInputs(1, INT_MAX)
  .NumOutputs(1)
  .IdenticalTypeOfInput(0)
  .CostInferenceFunction([](const OperatorDef& def,
                            const std::vector<TensorShape>& input_shape,
                            const std::vector<DataType>& input_type,
                            const std::vector<TensorShape>& output_shape,
                            const std::vector<DataType>& output_type) {
    OpSchema::Cost cost;
    for (size_t k = 0; k < input_shape.size(); ++k) {
      const TensorShape& shape = input_shape[k];
      size_t size = 1;
      for (auto dim : shape.dims()) size *= dim;
      cost.bytes_read = size * DataTypeSize(input_type[k]);
    }
    for (size_t k = 0; k < output_shape.size(); ++k) {
      const TensorShape& shape = output_shape[k];
      size_t size = 1;
      for (auto dim : shape.dims()) size *= dim;
      cost.bytes_written += size * DataTypeSize(output_type[k]);
    }
    return cost;
  })
  .SetDoc(R"DOC(
Concatenate a list of tensors into a single tensor
  )DOC");

}  // namespace blaze


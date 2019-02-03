/*
 * \file gather_op.cc
 * \brief The gather operation on cpu
 */
#include "blaze/operator/op/gather_op.h"

namespace blaze {

template <typename ValueType, typename IDType>
void RunGather(GatherParam<ValueType, IDType>& params) {
  size_t outer_size = params.y_size / params.y_inner_size;
  size_t indices_size = params.y_inner_size / params.inner_size;
  for (size_t i = 0; i < outer_size; ++i) {
    size_t src_offset = i * params.axis_size * params.inner_size;
    size_t dst_offset = i * params.y_inner_size;
    for (size_t j = 0; j < indices_size; ++j) {
      memcpy(params.y + dst_offset + j * params.inner_size,
             params.data + src_offset + params.indices[j] * params.inner_size,
             params.inner_size * sizeof(ValueType));
    }
  }
}

template <>
bool GatherOp<CPUContext>::RunOnDevice() {
  // Check the validity of Split
  CheckValid();

  Blob* data = this->Input(0);
  Blob* indices = this->Input(1);
  TYPE_SWITCH(data->data_type(), ValueType, {
  ID_TYPE_SWITCH(indices->data_type(), IDType, {

  GatherParam<ValueType, IDType> params;
  // Prepare params and reshape
  Setup<ValueType, IDType>(&params);
  // Start to execute gather kernel
  RunGather(params);

  });
  });
  return true;
}

REGISTER_CPU_OPERATOR(Gather, GatherOp<CPUContext>);

// Input: data indices Output: y
OPERATOR_SCHEMA(Gather)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
Gather a tensor base on indices.
    )DOC");

}  // namespace blaze


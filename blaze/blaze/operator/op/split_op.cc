/*
 * \file split_op.cc
 * \brief The split operation on cpu
 */
#include "blaze/operator/op/split_op.h"

namespace blaze {

template <typename DType>
void RunSplit(SplitParam<DType>& params) {
  size_t y_offset[kMaxInputSize] = { 0 };
  for (size_t i = 0; i < params.outer_size; ++i) {
    for (size_t k = 0; k < params.split_num; ++k) {
      memcpy(params.split_item[k].y + y_offset[k],
             params.x,
             params.split_item[k].split_axis_size * sizeof(DType));
      params.x += params.split_item[k].split_axis_size;
      y_offset[k] += params.split_item[k].split_axis_size;
    }
  }
}

template <>
bool SplitOp<CPUContext>::RunOnDevice() {
  if (auto_split_ && axis_ == 0) {
    return RunOnDevice_SplitAxis0();
  }
  // Check the validity of Split
  CheckValid();

  Blob* x = this->Input(0);
  TYPE_SWITCH(x->data_type(), DType, {
  
  SplitParam<DType> params;
  // Prepare params and reshape
  Setup<DType>(&params);
  // Start to execute split kernel
  RunSplit(params);

  });
  return true;
}

REGISTER_CPU_OPERATOR(Split, SplitOp<CPUContext>);

// Input: X Output: Y1, Y2, ...
OPERATOR_SCHEMA(Split)
    .NumInputs(1)
    .NumOutputs(1, INT_MAX)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
Split a tensor into many tensors.
    )DOC");

}  // namespace blaze


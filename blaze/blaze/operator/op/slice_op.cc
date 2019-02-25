/*
 * \file slice_op.cc 
 * \brief The slice operation
 */
#include "blaze/operator/op/slice_op.h"

namespace blaze {

template <typename DType>
void RunSlice(SliceParam<DType>& params) {
  size_t outer_size = params.sci.outer_size;
  size_t inner_size = params.sci.inner_size;
  size_t size = params.sci.size;
  size_t start = params.sci.start;
  size_t end = params.sci.end;

  size_t x_offset = start * inner_size;
  size_t x_inc = size * inner_size;
  size_t y_offset = 0;
  size_t y_inc = (end - start) * inner_size;
  size_t len = (end - start) * inner_size * sizeof(DType);

  for (size_t i = 0; i < outer_size; ++i) {
    memcpy(params.y + y_offset, params.x + x_offset, len);
    x_offset += x_inc;
    y_offset += y_inc;
  }
}

template <>
bool SliceOp<CPUContext>::RunOnDevice() {
  Blob* X = this->Input(0);

  // check the validity of SliceOp
  CheckValid();

  TYPE_SWITCH(X->data_type(), DType, {

  SliceParam<DType> params;
  Setup<DType>(&params);

  RunSlice<DType>(params);
  
  });  // TYPE_SWITCH(X->data_type(), DType, {
  return true;
}

REGISTER_CPU_OPERATOR(Slice, SliceOp<CPUContext>);

// Input: X Output: Y
// Arguments: axis, starts, ends
OPERATOR_SCHEMA(Slice)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeOfInput(0)
  .SetDoc(R"DOC(
Procuce a slice of the input tensor along one axis
  )DOC")
  .Input(0, "X", "N-D input tensor")
  .Output(0, "Y", "N-D output tensor");


}  // namespace blaze

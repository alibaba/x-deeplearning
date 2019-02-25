/*
 * \file fused_slice_concat_op.cc 
 * \brief The fused slice and concat operation
 */
#include "blaze/operator/fused_op/fused_slice_concat_op.h"

namespace blaze {

template <typename DType>
void RunSliceConcat(FusedSliceConcatParam<DType>& params) {
  if (params.slice_axis == params.concat_axis)  {
    size_t y_offset = 0;
    size_t x_inc = params.slice_axis_size * params.slice_inner_size;
    for (size_t i = 0; i < params.concat_input_size; ++i) {
      params.slice_item[i].start *= params.slice_inner_size;
      params.slice_item[i].end *= params.slice_inner_size;
    }
    for (size_t i = 0; i < params.concat_outer_size; ++i) {
      for (size_t k = 0; k < params.concat_input_size; ++k) {
        memcpy(params.y + y_offset, params.x + params.slice_item[k].start,
               (params.slice_item[k].end - params.slice_item[k].start) * sizeof(DType));
        y_offset += (params.slice_item[k].end - params.slice_item[k].start);
      }
      params.x += x_inc;
    }
  } else if (params.slice_axis < params.concat_axis) {
    size_t span = ((params.slice_item[0].end - params.slice_item[0].start) * params.slice_inner_size) /
        (params.concat_axis_size / params.concat_input_size * params.concat_inner_size);
    size_t y_offset = 0;
    size_t x_inc = params.slice_axis_size * params.slice_inner_size;
    size_t x_n = params.concat_inner_size * params.concat_axis_size / params.concat_input_size;

    for (size_t i = 0; i < params.slice_outer_size; ++i) {
      for (size_t j = 0; j < span; ++j) {
        for (size_t k = 0; k < params.concat_input_size; ++k) {
          DType* x = params.x + i * x_inc + params.slice_item[k].start * params.slice_inner_size + j * x_n;
          memcpy(params.y + y_offset, x, x_n * sizeof(DType));
          y_offset += x_n;
        }
      }
    }
  } else {
    size_t span = (params.concat_axis_size / params.concat_input_size * params.concat_inner_size) / 
        ((params.slice_item[0].end - params.slice_item[0].start) * params.slice_inner_size);
    size_t y_offset = 0;
    size_t x_outer_inc = (params.slice_outer_size * params.slice_axis_size * params.slice_inner_size) /
        (params.concat_outer_size);
    size_t x_inc = params.slice_axis_size * params.slice_inner_size;

    for (size_t i = 0; i < params.concat_input_size; ++i) {
      params.slice_item[i].start *= params.slice_inner_size;
      params.slice_item[i].end *= params.slice_inner_size;
    }

    for (size_t i = 0; i < params.concat_outer_size; ++i) {
      for (size_t j = 0; j < params.concat_input_size; ++j) { 
        for (size_t k = 0; k < span; ++k) {
          DType* x = params.x + i * x_outer_inc + k * x_inc + params.slice_item[j].start;
          memcpy(params.y + y_offset, x, (params.slice_item[j].end - params.slice_item[j].start) * sizeof(DType));
          y_offset += params.slice_item[j].end - params.slice_item[j].start;
        }
      }
    }
  }
}

template <>
bool FusedSliceConcatOp<CPUContext>::RunOnDevice() {
  // check the validity of FusedSliceConcat
  CheckValid();

  Blob* x0 = this->Input(0);
  TYPE_SWITCH(x0->data_type(), DType, {
  
  FusedSliceConcatParam<DType> params;
  // Prepare params and reshape
  Setup<DType>(&params);

  // Start to execute slice/concat fused kernel
  RunSliceConcat(params);

  });
  return true;
}

REGISTER_CPU_OPERATOR(FusedSliceConcat, FusedSliceConcatOp<CPUContext>);

// Input: X Output: Y
OPERATOR_SCHEMA(FusedSliceConcat)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
FusedSliceConcat take mutiple slice and one concat operations into one operation.
  )DOC")
  .Input(0, "X", "N-D Input tensor")
  .Output(0, "Y", "N-D output tensor");

}  // namespace blaze


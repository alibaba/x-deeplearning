/*
 * \file fuse_op.cc 
 * \brief The fuse operation
 */
#include "blaze/operator/op/fuse_op.h"

namespace blaze {

template <typename DType>
void FuseKernel(const FuseParam<DType>& params) {
  size_t offset = 0;
  size_t x1_offset = 0, x2_offset = 0;
  size_t x1_inc, x2_inc;
  if (params.x1_m == 1) {
    x1_inc = 0;
    x2_inc = params.x2_n;
  } else {
    x1_inc = params.x1_n;
    x2_inc = 0;
  }
  for (size_t row = 0; row < params.y_m; ++row) {
    memcpy(params.y + offset, params.x1 + x1_offset, params.x1_n  * sizeof(DType));
    x1_offset += x1_inc;
    offset += params.x1_n;

    memcpy(params.y + offset, params.x2 + x2_offset, params.x2_n * sizeof(DType));
    x2_offset += x2_inc;
    offset += params.x2_n;
  }
}

template <>
bool FuseOp<CPUContext>::RunOnDevice() {
  CheckValid();
  Blob* x1 = this->Input(0);
  TYPE_SWITCH(x1->data_type(), DType, {
    FuseParam<DType> params;
    Setup<DType>(&params);
    FuseKernel(params);
  });
  return true;
}

REGISTER_CPU_OPERATOR(Fuse, FuseOp<CPUContext>);

// Input: X, W, R, B Output: Y
OPERATOR_SCHEMA(Fuse)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
 fuse operation.
    )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze

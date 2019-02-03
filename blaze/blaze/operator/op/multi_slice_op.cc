/*
 * \file multi_slice_op.cc 
 * \brief The multi slice operation
 */
#include "blaze/operator/op/multi_slice_op.h"

namespace blaze {

template <>
void MultiSliceOp<CPUContext>::Memcpy2D(void* dst, size_t dpitch,
    const void* src, size_t spitch,
    size_t width, size_t height) const {
  char* cdst = static_cast<char *>(dst);
  const char* csrc = static_cast<const char *>(src);
  for (size_t i = 0; i < height; ++i) {
    memcpy(cdst, csrc, width);
    cdst += dpitch;
    csrc += spitch;
  }
}

template <>
bool MultiSliceOp<CPUContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  TYPE_SWITCH(x->data_type(), DType, {
    MultiSliceParam<DType> param;
    Setup(&param);
    // copy data 
    if (use_memcpy2d_) {
      SliceMemcpy2D<DType>();
    } else {
      MultiSliceMemcpy<DType>(); 
    }
  });
   
  return true;
}

REGISTER_CPU_OPERATOR(MultiSlice, MultiSliceOp<CPUContext>);

// Input: X, W, R, B Output: Y
OPERATOR_SCHEMA(MultiSlice)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeOfInput(0)
    .SetDoc(R"DOC(
Multi slice operation.
    )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");

}  // namespace blaze

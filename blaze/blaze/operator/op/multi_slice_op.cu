/*
 * \file multi_slice_op.cu 
 * \brief The multi slice operation
 */
#include "blaze/operator/op/multi_slice_op.h"

#include "blaze/common/common_defines.h"

namespace blaze {

template <>
void MultiSliceOp<CUDAContext>::Memcpy2D(void* dst, size_t dpitch,
    const void* src, size_t spitch,
    size_t width, size_t height) const {
  cudaStream_t stream = this->context_.cuda_stream();
  CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch,
        width, height, cudaMemcpyDefault, stream)); 
}

template <>
bool MultiSliceOp<CUDAContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  TYPE_SWITCH_ON_CUDA(x->data_type(), DType, {
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

REGISTER_CUDA_OPERATOR(MultiSlice, MultiSliceOp<CUDAContext>);

}  // namespace blaze

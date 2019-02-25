/*
 * \file cudnn_utils.cc
 * \brief The cudnn utils routine.
 */
#ifdef USE_CUDA

#include "blaze/common/types.h"
#include "blaze/math/cudnn_utils.h"

namespace blaze {

#ifndef DECLARE_SET_TENSOR_4D_DESC
#define DECLARE_SET_TENSOR_4D_DESC(DType, CUDNN_DType)               \
  template <>                                                        \
  void CUDNNSetTensor4dDesc<DType>(cudnnTensorDescriptor_t* desc,    \
                                   int n,                            \
                                   int c,                            \
                                   int h,                            \
                                   int w) {                          \
    const int stride_w = 1;                                          \
    const int stride_h = w * stride_w;                               \
    const int stride_c = h * stride_h;                               \
    const int stride_n = c * stride_c;                               \
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, CUDNN_DType,     \
                                 n, c, h, w, stride_n, stride_c,     \
                                 stride_h, stride_w));               \
  }
#endif

DECLARE_SET_TENSOR_4D_DESC(float16, CUDNN_DATA_HALF)
DECLARE_SET_TENSOR_4D_DESC(float, CUDNN_DATA_FLOAT)
DECLARE_SET_TENSOR_4D_DESC(double, CUDNN_DATA_DOUBLE)

#undef DECLARE_SET_TENSOR_4D_SESC

}  // blaze

#endif

/*
 * \file cudnn_utils.h
 * \brief The cudnn utils routine.
 */
#pragma once

#include "blaze/common/common_defines.h"

#include "blaze/common/exception.h"

namespace blaze {

template <typename DType>
void CUDNNSetTensor4dDesc(cudnnTensorDescriptor_t* desc,
                          int n,
                          int c,
                          int h,
                          int w);

template <typename DType>
void CUDNNSoftmaxForward(cudnnHandle_t handle,
                         cudnnTensorDescriptor_t* x_desc,
                         const DType* x,
                         cudnnTensorDescriptor_t* y_desc,
                         DType* y) {
  static DType kOne = 1.0, kZero = 0.0;
  CUDNN_CHECK(cudnnSoftmaxForward(handle,
                                  CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &kOne,
                                  *x_desc,
                                  x,
                                  &kZero,
                                  *y_desc,
                                  y));
}

}  // namespace blaze


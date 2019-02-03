/*
 * \file softmax_op.cu
 * \brief The softmax operation on GPU
 */
#include "blaze/operator/op/softmax_op.h"

#include "blaze/math/cudnn_utils.h"
#include "blaze/math/vml.h"

namespace blaze {

template <>
SoftmaxOp<CUDAContext>::SoftmaxOp(const OperatorDef& def, Workspace* workspace) :
    Operator<CUDAContext>(def, workspace) {
  axis_ = OperatorBase::GetSingleArgument<size_t>("axis", 1);

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
  
  // if the input is not fp32, the input data should be converted to float32
  x_fp32_.reset(new Blob(this->device_option_));
  y_fp32_.reset(new Blob(this->device_option_));
}

template <>
bool SoftmaxOp<CUDAContext>::RunOnDevice() {
  Blob* X = this->Input(0);
  Blob* Y = this->Output(0);

  TYPE_SWITCH_ON_CUDA(X->data_type(), DType, {
  
  // Reshape
  Y->Reshape(X->shape());

  int N = X->size(0, axis_);
  int C = X->shape()[axis_];
  int H = X->size(axis_ + 1, X->shape().size());
  int W = 1;

  float *cx, *cy;
  if (X->data_type() != TypeFlag<float>::kFlag) {
    x_fp32_->Reshape(X->shape());
    y_fp32_->Reshape(Y->shape());

    VML_Set<float, DType>(X->size(), x_fp32_->as<float>(), X->as<DType>(), &this->context_);

    cx = x_fp32_->as<float>();
    cy = y_fp32_->as<float>();
  } else {
    cx = X->as<float>();
    cy = Y->as<float>();
  }

  CUDNNSetTensor4dDesc<float>(&bottom_desc_, N, C, H, W);
  CUDNNSetTensor4dDesc<float>(&top_desc_, N, C, H, W);

  cudnnHandle_t handle = this->context_.cudnn_handle();
  CUDNNSoftmaxForward(handle, &bottom_desc_, cx, &top_desc_, cy);
  
  if (X->data_type() != TypeFlag<float>::kFlag) {
    VML_Set<DType, float>(Y->size(), Y->as<DType>(), y_fp32_->as<float>(), &this->context_);
  }

  });
  
  return true;
}

REGISTER_CUDA_OPERATOR(Softmax, SoftmaxOp<CUDAContext>);

}  // namespace blaze


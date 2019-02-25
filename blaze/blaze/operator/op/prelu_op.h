/*
 * \file prelu_op.h
 * \desc The prelu operator.
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType, typename WDType>
struct PReluParam {
  DType* x;
  WDType* w;
  DType* y;
  size_t size;
  size_t inner_size;

  PReluParam(DType* x, WDType* w, DType* y, size_t size, size_t inner_size) :
      x(x), w(w), y(y), size(size), inner_size(inner_size) { }
};

template <class Context>
class PReluOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  PReluOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) { } 

  bool RunOnDevice() override;

 protected:
  // Check the validity of op inputs 
  void CheckValid() {
    Blob* X = this->Input(0);
    Blob* W = this->Input(1);
    Blob* Y = this->Output(0);

    BLAZE_CONDITION_THROW(X->device_type() == W->device_type(),
                          "X->device_type()=",
                          X->device_type(),
                          " W->device_type()=",
                          W->device_type());
    BLAZE_CONDITION_THROW(Context::device_type() == X->device_type(),
                          "X->device_type()=",
                          X->device_type(),
                          " is not ",
                          Context::device_type());
    BLAZE_CONDITION_THROW(X->data_type() == Y->data_type(),
                          "X->data_type()=",
                          X->data_type(),
                          " Y->data_type()=",
                          Y->data_type());

    BLAZE_CONDITION_THROW(W->shape().size() <= X->shape().size(),
                          "W->shape().size()=",
                          W->shape().size(),
                          " X->shape().size()=",
                          X->shape().size());
  }
};

}  // namespace blaze

/*
 * \file concat_op.h 
 * \brief The concat operation
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/operator/common_helper.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
struct ConcatItem {
  DType* x;
  size_t axis_size;
};

template <typename DType>
struct ConcatParam {
  ConcatItem<DType> concat_item[kMaxInputSize];
  size_t input_size;
  size_t axis;
  size_t outer_size;
  size_t inner_size;
  DType* y;
  size_t y_size;
  size_t concat_axis_size;
};

template <class Context>
class ConcatOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  ConcatOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    axis_ = OperatorBase::GetSingleArgument<int>("axis", 1);
    input_size_ = def.input_size();
    // Note: the max input size must <= 32
    BLAZE_CONDITION_THROW(input_size_ <= kMaxInputSize, "input_size_=", input_size_);
  }

  bool RunOnDevice() override;

 protected:
  // Check the context device type and data type validity
  void CheckValid() {
    BLAZE_CONDITION_THROW(this->input_size_ > 0, "input_size equals zero");

    Blob* x0 = this->Input(0);
    BLAZE_CONDITION_THROW(x0->device_type() == Context::device_type(),
                          "x0->device_type()=",
                          x0->device_type(),
                          " is not ",
                          Context::device_type());
    const std::vector<size_t>& shape0 = x0->shape();
    BLAZE_CONDITION_THROW(axis_ < shape0.size(),
                          "axis_=",
                          axis_,
                          " shape0.size()=",
                          shape0.size(), " ", this->def_.name());

    for (size_t k = 1; k < this->input_size_; ++k) {
      Blob* xk = this->Input(k);

      BLAZE_CONDITION_THROW(x0->device_type() == xk->device_type(),
                            "x0->device_type()=",
                            x0->device_type(),
                            " xk->device_type()=",
                            xk->device_type());
      BLAZE_CONDITION_THROW(x0->data_type() == xk->data_type(),
                            "x0->data_type()=",
                            x0->data_type(),
                            " xk->data_type()=",
                            xk->data_type());

      // NOTE: not support unibroadcast concat now.
      const std::vector<size_t>& shapek = xk->shape();
      BLAZE_CONDITION_THROW(shapek.size() == shape0.size(),
                            "shape0.size()=",
                            shape0.size(),
                            " shapek.size()=",
                            shapek.size(), " k=", k, " op=", this->def_.DebugString());

      for (int i = 0; i < shape0.size(); ++i) {
        if (i == axis_) continue;
        BLAZE_CONDITION_THROW(shape0[i] == shapek[i],
                              "i=", i, " shape0[i]=", shape0[i], " shapek[i]=", shapek[i],
                              " k=", k,
                              " op=", this->def_.DebugString());
      }
    }
  }

  // Prepare concat param
  template <typename DType>
  void Setup(ConcatParam<DType>* params) {
    const std::vector<size_t>& x0_shape = this->Input(0)->shape();
    Blob* y = this->Output(0);

    params->input_size = this->input_size_;
    params->axis = this->axis_;
    params->outer_size = 1;
    for (size_t k = 0; k < this->axis_; ++k) params->outer_size *= x0_shape[k];
    params->inner_size = 1;
    for (size_t k = this->axis_ + 1; k < x0_shape.size(); ++k) params->inner_size *= x0_shape[k];

    size_t concat_axis_size = 0;
    for (size_t k = 0; k < this->input_size_; ++k) {
      Blob* blob = this->Input(k);
      params->concat_item[k].axis_size = blob->shape()[this->axis_] * params->inner_size;
      params->concat_item[k].x = blob->as<DType>();
      concat_axis_size += blob->shape()[this->axis_];
    }
    std::vector<size_t> y_shape = x0_shape;
    y_shape[this->axis_] = concat_axis_size;

    // Reshape y.
    y->Reshape(y_shape);
    params->y = y->as<DType>();
    params->y_size = y->size();
    params->concat_axis_size = concat_axis_size;
  }

  size_t axis_;
  size_t input_size_;
};

}  // namespace blaze

/*
 * \file split_op.h 
 * \brief The split operation
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/operator/common_helper.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename DType>
struct SplitItem {
  DType* y;
  size_t split_axis_offset;
  size_t split_axis_size;
};

template <typename DType>
struct SplitParam {
  DType* x;
  size_t outer_size;
  size_t axis;
  size_t inner_size;
  size_t axis_size;
  size_t split_num;
  SplitItem<DType> split_item[kMaxInputSize];
};

template <typename Context>
class SplitOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  SplitOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    axis_ = OperatorBase::GetSingleArgument<size_t>("axis", 1);
    if (OperatorBase::HasArgument("split")) {
      auto_split_ = false;
      splits_ = OperatorBase::GetRepeatedArgument<size_t>("split");
      BLAZE_CONDITION_THROW(splits_.size() == this->OutputSize(),
                            "splits_.size()=", splits_.size(),
                            " this->output_size()=", this->OutputSize());
    } else {
      auto_split_ = true;
    }
  }

  bool RunOnDevice() override;

 protected:
  // RunOnDevice when split on Axis-0
  bool RunOnDevice_SplitAxis0() {
    Blob* x = this->Input(0);
    std::vector<TIndex> shape = x->shape();
    BLAZE_CONDITION_THROW(shape[0] % this->OutputSize() == 0, "shape[0]=%u", shape[0]);
    shape[0] /= this->OutputSize();
    TIndex x_size = x->size() / this->OutputSize();
    for (int i = 0; i < this->OutputSize(); ++i) {
      Blob* y = this->Output(i);
      TYPE_SWITCH_WITH_CTX(this->context_, x->data_type(), DType, {
        y->RefReshape(shape, x->as<DType>() + x_size * i);
      });
    }
    return true;
  }

  // Check the context device type and data type validity
  void CheckValid() {
    Blob* x0 = this->Input(0);
    size_t output_size = this->OutputSize();
    const std::vector<size_t>& shape0 = x0->shape();
    BLAZE_CONDITION_THROW(axis_ < shape0.size(), "axis_=", axis_, " shape0.size()=", shape0.size());
    if (splits_.empty()) {
      BLAZE_CONDITION_THROW(shape0[axis_] % output_size == 0,
                            "shape0[", axis_, "]=", shape0[axis_],
                            " output_size=", output_size, " op=", this->def_.name());
    } else {
      BLAZE_CONDITION_THROW(output_size == splits_.size(),
                            "output_size=", output_size,
                            " splits_.size()=", splits_.size());
      size_t sum_axis = 0;
      for (auto sp : splits_) sum_axis += sp;
      BLAZE_CONDITION_THROW(sum_axis == shape0[axis_],
                            "sum_axis=", sum_axis,
                            " shape0[", axis_, "]=", shape0[axis_]);
    }
  }

  // Prepare split param
  template <typename DType>
  void Setup(SplitParam<DType>* params) {
    params->split_num = this->OutputSize();
    Blob* x = this->Input(0);
    std::vector<size_t> shape = x->shape();
    
    params->x = x->as<DType>();
    params->axis = axis_;
    params->outer_size = 1;
    for (size_t i = 0; i < axis_; ++i) params->outer_size *= shape[i];
    params->inner_size = 1;
    for (size_t i = axis_ + 1; i < shape.size(); ++i) params->inner_size *= shape[i];
    params->axis_size = shape[axis_];

    size_t offset = 0;
    size_t avg = params->axis_size / params->split_num;
    for (size_t k = 0; k < params->split_num; ++k) {
      shape[axis_] = splits_.empty() ? avg : splits_[k];
      Blob* y = this->Output(k);
      y->Reshape(shape);

      params->split_item[k].y = y->as<DType>();
      params->split_item[k].split_axis_offset = offset;
      params->split_item[k].split_axis_size = shape[axis_] * params->inner_size;
      offset += shape[axis_];
    }
  }

  size_t axis_;
  std::vector<size_t> splits_;
  bool auto_split_;
};

}  // namespace blaze


/*
 * \file fused_slice_concat_op.h 
 * \brief The fused slice and concat operation
 *
 * Such as:
 *
 *          Input
 *      |      |      |
 *      |      |      |
 *      Slice  Slice  Slice
 *       \      |     /
 *        \     |    /
 *           Concat(Output)
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/operator/common_helper.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

struct SliceItem {
  size_t start;
  size_t end;
  size_t len;
};

template <typename DType>
struct FusedSliceConcatParam {
  DType* x;

  size_t slice_axis;
  size_t slice_axis_size;
  size_t slice_outer_size;
  size_t slice_inner_size;
  SliceItem slice_item[kMaxInputSize];

  size_t concat_input_size;
  size_t concat_axis;
  size_t concat_outer_size;
  size_t concat_inner_size;
  size_t concat_axis_size;

  DType* y;
  size_t y_size;
};

template <class Context>
class FusedSliceConcatOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  FusedSliceConcatOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    concat_axis_ = OperatorBase::GetSingleArgument<size_t>("concat_axis", 0);
    slice_axis_ = OperatorBase::GetSingleArgument<size_t>("slice_axis", 0);
    start_ = OperatorBase::GetRepeatedArgument<size_t>("start");
    end_ = OperatorBase::GetRepeatedArgument<size_t>("end");

    BLAZE_CONDITION_THROW(start_.size() == end_.size() && start_.size() > 0,
                          "start_.size()=", start_.size(),
                          " end_.size()=", end_.size());
    if (concat_axis_ != slice_axis_) {
      // if the slice and concat axis not equal, the slice segment must be
      // equal.
      size_t dim = end_[0] - start_[0];
      for (size_t k = 1; k < this->start_.size(); ++k) {
        BLAZE_CONDITION_THROW(dim == end_[k] - start_[k],
                              "dim=", dim,
                              " start[", k,
                              "]=", start_[k],
                              " end[", k,
                              "]=", end_[k]);
      }
    }
  }

  bool RunOnDevice() override;

 protected:
  // Prepare slice/concat fused param
  template <typename DType>
  void Setup(FusedSliceConcatParam<DType>* params) {
    params->slice_axis = slice_axis_;
    params->concat_axis = concat_axis_;

    Blob* x = this->Input(0);
    Blob* y = this->Output(0);

    // Set slice params
    const std::vector<size_t>& x_shape = x->shape();
    params->slice_outer_size = 1;
    for (size_t k = 0; k < slice_axis_; ++k) {
      params->slice_outer_size *= x_shape[k];
    }
    params->slice_axis_size = x_shape[slice_axis_];
    params->slice_inner_size = 1;
    for (size_t k = slice_axis_ + 1; k < x_shape.size(); ++k) {
      params->slice_inner_size *= x_shape[k];
    }
    for (size_t i = 0; i < start_.size(); ++i) {
      params->slice_item[i].start = start_[i];
      params->slice_item[i].end = end_[i];
      params->slice_item[i].len = end_[i] - start_[i];
    }

    // Set concat params
    params->concat_input_size = start_.size();

    std::vector<size_t> y_shape = x_shape;
    if (concat_axis_ != slice_axis_) {
      y_shape[slice_axis_] = end_[0] - start_[0];
      y_shape[concat_axis_] *= params->concat_input_size;
    } else {
      y_shape[concat_axis_] = 0;
      for (size_t i = 0; i < start_.size(); ++i) {
        y_shape[concat_axis_] += end_[i] - start_[i];
      }
    }

    params->concat_outer_size = 1;
    for (size_t k = 0; k < concat_axis_; ++k) {
      params->concat_outer_size *= y_shape[k];
    }
    params->concat_axis_size = y_shape[concat_axis_];
    params->concat_inner_size = 1;
    for (size_t k = concat_axis_ + 1; k < y_shape.size(); ++k) {
      params->concat_inner_size *= y_shape[k];
    }

    params->x = x->as<DType>();
    // Reshape for larger space.
    y->Reshape(y_shape);
    params->y = y->as<DType>();
    params->y_size = y->size();
  }

  // Check the validity with input
  void CheckValid() {
    const auto& x_shape = this->Input(0)->shape();
    BLAZE_CONDITION_THROW(slice_axis_ >= 0 && slice_axis_ < x_shape.size(),
                          "slice_axis_=", slice_axis_,
                          " x_shape.size()=", x_shape.size());
    BLAZE_CONDITION_THROW(concat_axis_ >= 0 && concat_axis_ < x_shape.size(),
                          "concat_axis_=", concat_axis_,
                          " x_shape.size()=", x_shape.size());

    for (size_t k = 0; k < this->start_.size(); ++k) {
      BLAZE_CONDITION_THROW(start_[k] >= 0 && start_[k] < x_shape[slice_axis_],
                            "start[", k, "]=", start_[k],
                            " x_shape[", slice_axis_, "]=", x_shape[slice_axis_]);
      BLAZE_CONDITION_THROW(end_[k] >= 0 && end_[k] <= x_shape[slice_axis_],
                            "end[", k, "]=", end_[k],
                            " x_shape[", slice_axis_, "]=", x_shape[slice_axis_]);
    }
  }

  std::vector<size_t> start_;
  std::vector<size_t> end_;
  size_t concat_axis_;
  size_t slice_axis_;
};

}  // namespace blaze


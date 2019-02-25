/*
 * \file slice_op.h 
 * \brief The slice operation
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

struct SliceCopyInfo {
  size_t outer_size;
  size_t inner_size;
  size_t axis;
  size_t size;
  size_t start;
  size_t end;
};

template <typename DType>
struct SliceParam {
  SliceCopyInfo sci;
  DType* x;
  DType* y;
};

template <class Context>
class SliceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  SliceOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    // Unlike ONNX, Blaze only support one dimentional slice
    if (OperatorBase::HasArgument("axes")) {
      std::vector<size_t> axes = OperatorBase::GetRepeatedArgument<size_t>("axes");
      BLAZE_CONDITION_THROW(axes.size() == 1, "axes.size()=", axes.size());
      axis_ = axes[0];
    } else {
      axis_ = OperatorBase::GetSingleArgument<size_t>("axis", 0);
    }

    if (OperatorBase::HasArgument("starts")) {
      std::vector<size_t> starts = OperatorBase::GetRepeatedArgument<size_t>("starts");
      std::vector<size_t> ends = OperatorBase::GetRepeatedArgument<size_t>("ends");
      BLAZE_CONDITION_THROW(starts.size() == 1 && ends.size() == 1,
                            "starts.size()=", starts.size(),
                            " ends.size()=", ends.size());
      start_ = starts[0];
      end_ = ends[0];
    } else {
      start_ = OperatorBase::GetSingleArgument<size_t>("start", 0);
      end_ = OperatorBase::GetSingleArgument<size_t>("end", 1);
    }
    BLAZE_CONDITION_THROW(start_ < end_,
                          "Slice start=",
                          start_,
                          " end=",
                          end_);
  }

  bool RunOnDevice() override;

 protected:
  // Setup
  template <typename DType>
  void Setup(SliceParam<DType>* params) {
    Blob* X = this->Input(0);
    Blob* Y = this->Output(0);
    const std::vector<size_t>& shape = X->shape();
   
    // Only one dimension slice
    params->sci.axis = axis_;
    params->sci.size = shape[params->sci.axis];
    params->sci.start = start_;
    params->sci.end = end_;
    params->sci.outer_size = 1;
    for (size_t z = 0; z < axis_; ++z) {
      params->sci.outer_size *= shape[z];
    }
    params->sci.inner_size = 1;
    for (size_t z = axis_ + 1; z < shape.size(); ++z) {
      params->sci.inner_size *= shape[z];
    }
    LOG_DEBUG("axis=%u size=%u start=%u end=%u outer_size=%u inner_size=%u",
              params->sci.axis,
              params->sci.size,
              params->sci.start,
              params->sci.end,
              params->sci.outer_size,
              params->sci.inner_size);

    params->x = X->as<DType>();
    std::vector<size_t> y_shape = X->shape();
    y_shape[params->sci.axis] = params->sci.end - params->sci.start;

    // Reshape
    Y->Reshape(y_shape);
    
    params->y = Y->as<DType>();
  }

  // Check the context device type and data type validity
  void CheckValid() {
    Blob* X = this->Input(0);
    Blob* Y = this->Output(0);
    
    BLAZE_CONDITION_THROW(X->device_type() == Y->device_type(),
                          "X->device_type()=",
                          X->device_type(),
                          " Y->device_type()=",
                          Y->device_type());
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

    const std::vector<size_t>& shape = X->shape();
    BLAZE_CONDITION_THROW(axis_ < shape.size(),
                          "axis_=",
                          axis_,
                          " shape.size()=",
                          shape.size(), " ", this->def_.name());

    BLAZE_CONDITION_THROW(axis_ < shape.size() &&
                          end_ <= shape[axis_],
                          "axis=",
                          axis_,
                          " shape.size()=",
                          shape.size(),
                          " end_=",
                          end_,
                          " shape[exes_]=",
                          shape[axis_]);
  }

  size_t axis_;
  size_t start_;
  size_t end_;
};

}  // namespace blaze


/*
 * \file gather_op.h 
 * \brief The gather operation
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename ValueType, typename IDType>
struct GatherParam {
  ValueType* data;
  size_t axis_size;
  size_t inner_size;
  IDType* indices;
  ValueType* y;
  size_t y_inner_size;
  size_t y_size;
};

template <typename Context>
class GatherOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  GatherOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    axis_ = OperatorBase::GetSingleArgument<size_t>("axis", 0);
  }

  bool RunOnDevice() override;

 protected:
  // Check the context device type and data type validity
  void CheckValid() {
    Blob* data = this->Input(0);
    Blob* indices = this->Input(1);
    BLAZE_CONDITION_THROW(IsIntegerType(indices->data_type()), "indices->data_type()=",
                          indices->data_type());
  }

  template <typename ValueType, typename IDType>
  void Setup(GatherParam<ValueType, IDType>* params) {
    Blob* data = this->Input(0);
    Blob* indices = this->Input(1);
    Blob* output = this->Output(0);

    const std::vector<size_t>& data_shape = data->shape();
    const std::vector<size_t>& indices_shape = indices->shape();

    std::vector<size_t> o_shape;
    size_t axis_size = data_shape[axis_];
    size_t inner_size = 1;
    size_t y_inner_size = 1;
    for (size_t i = 0; i < axis_; ++i) {
      o_shape.push_back(data_shape[i]);
    }
    for (const auto& item : indices_shape) {
      o_shape.push_back(item);
      y_inner_size *= item;
    }
    for (size_t i = axis_ + 1; i < data_shape.size(); ++i) {
      o_shape.push_back(data_shape[i]);
      inner_size *= data_shape[i];
      y_inner_size *= data_shape[i];
    }
    output->Reshape(o_shape);

    // set params arguments
    params->data = data->as<ValueType>();
    params->axis_size = axis_size;
    params->inner_size = inner_size;
    params->indices = indices->as<IDType>();
    params->y = output->as<ValueType>();
    params->y_inner_size = y_inner_size;
    params->y_size = output->size();
  }

  size_t axis_;
};

}  // namespace blaze

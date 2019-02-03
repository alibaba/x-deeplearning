/*
 * \file intersect_op.h 
 * \brief The intersect operator for feature intersect.
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <typename K_DType, typename V_DType, typename N_DType>
struct IntersectParam {
  K_DType* x_ids;
  V_DType* x_values;
  K_DType* y_ids;
  N_DType* y_nums;
  K_DType* z_ids;
  V_DType* z_values;
  N_DType* z_nums;
  size_t ad_num;
  size_t x_size;
  size_t y_size;
  Blob* z_id_blob;
  Blob* z_value_blob;
  Blob* z_num_blob;
};

template <class Context>
class IntersectOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  IntersectOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    need_sort_ = OperatorBase::GetSingleArgument<bool>("need_sort", true);
  }
  
  bool RunOnDevice() override;

 protected:
  // Prepare intersect param
  template <typename K_DType, typename V_DType, typename N_DType>
  void Setup(IntersectParam<K_DType, V_DType, N_DType>* params) {
    Blob* x_id_blob = this->Input(0);
    Blob* x_value_blob = this->Input(1);
    Blob* x_num_blob = this->Input(2);
    Blob* y_id_blob = this->Input(3);
    Blob* y_value_blob = this->Input(4);
    Blob* y_num_blob = this->Input(5);
    Blob* z_id_blob = this->Output(0);
    Blob* z_value_blob = this->Output(1);
    Blob* z_num_blob = this->Output(2);

    params->x_ids = x_id_blob->as<K_DType>();
    params->x_values = x_value_blob->as<V_DType>();
    params->y_ids = y_id_blob->as<K_DType>();
    params->y_nums = y_num_blob->as<N_DType>();
    params->x_size = x_id_blob->size();
    params->y_size = y_id_blob->size();

    params->z_id_blob = z_id_blob;
    params->z_value_blob = z_value_blob;
    params->z_num_blob = z_num_blob;

    // setup output blob shape
    const auto& y_id_shape = y_id_blob->shape();
    const auto& y_num_shape = y_num_blob->shape();
    params->z_id_blob->Reshape(y_id_shape);
    params->z_value_blob->Reshape(y_id_shape);
    params->z_num_blob->Reshape(y_num_shape);

    params->z_ids = z_id_blob->as<K_DType>();
    params->z_values = z_value_blob->as<V_DType>();
    params->z_nums = z_num_blob->as<N_DType>();

    params->ad_num = y_num_blob->size();
  }

  void CheckValid() {
    Blob* x_id = this->Input(0);
    Blob* x_value = this->Input(1);
    Blob* x_num = this->Input(2);
    Blob* y_id = this->Input(3);
    Blob* y_value = this->Input(4);
    Blob* y_num = this->Input(5);

    BLAZE_CONDITION_THROW(x_id->data_type() == y_id->data_type(),
                          "x_id->data_type()=",
                          x_id->data_type(),
                          " y_id->data_type()=",
                          y_id->data_type());
    BLAZE_CONDITION_THROW(x_value->data_type() == y_value->data_type(),
                          "x_value->data_type()=",
                          x_value->data_type(),
                          " y_value->data_type()=",
                          y_value->data_type());
    BLAZE_CONDITION_THROW(x_num->data_type() == y_num->data_type(),
                          "x_num->data_type()=",
                          x_id->data_type(),
                          " y_num->data_type()=",
                          y_id->data_type());

    // user feature check
    BLAZE_CONDITION_THROW(x_num->size() == 1,
                          "x_num->size()=",
                          x_num->size());

    // id & value size must be equal
    BLAZE_CONDITION_THROW(x_id->size() == x_value->size(),
                          "x_id->size()=",
                          x_id->size(),
                          " x_value->size()=",
                          x_value->size());
    BLAZE_CONDITION_THROW(y_id->size() == y_value->size(),
                          "y_id->size()=",
                          y_id->size(),
                          " y_value->size()=",
                          y_value->size());
  }

  bool need_sort_ = true;
};

}  // namespace blaze

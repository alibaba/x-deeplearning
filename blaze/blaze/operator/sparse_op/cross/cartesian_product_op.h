/*
 * \file cartesian_op.h
 * \brief The cartesian product operator for cartesian product combine feature.
 */
#pragma once

#include <vector>

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

#include "cross_common_defines.h"

namespace blaze {

template <typename K_DType, typename V_DType, typename N_DType>
struct CartesianItem {
  K_DType* ids;
  V_DType* values;
  N_DType* nums;
  size_t num_size;
  N_DType process_start;
  N_DType process_end;
};

template <typename K_DType, typename V_DType, typename N_DType>
struct CartesianProductParam {
  std::vector<CartesianItem<K_DType, V_DType, N_DType> > input_items;
  CartesianItem<K_DType, V_DType, N_DType> cartesian_output;
};

template <class Context>
class CartesianProductOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  CartesianProductOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
  }

  bool RunOnDevice() override;

 protected:
  // Prepare intersect param
  template <typename K_DType, typename V_DType, typename N_DType>
  void Setup(CartesianProductParam<K_DType, V_DType, N_DType>* params) {
    params->cartesian_output.num_size = 0;
    // prepare cartesian input item
    params->input_items.resize(this->InputSize() / 3);
    for (auto i = 0, p = 0; i < this->InputSize(); i += 3, ++p) {
      Blob* id_blob = this->Input(i);
      params->input_items[p].ids = id_blob->as<K_DType>();
      Blob* value_blob = this->Input(i + 1);
      params->input_items[p].values = value_blob->as<V_DType>();
      Blob* num_blob = this->Input(i + 2);
      params->input_items[p].nums = num_blob->as<N_DType>();
      params->input_items[p].num_size= num_blob->size();

      if (params->input_items[p].num_size > params->cartesian_output.num_size) {
        if (params->cartesian_output.num_size <= 1) {
          params->cartesian_output.num_size = params->input_items[p].num_size;
        } else {
          BLAZE_THROW("can't resolve multi input with different num size");
        }
      }
    }

    // prepare cartesian output
    Blob* output_id_blob = this->Output(0);
    Blob* output_value_blob = this->Output(1);
    Blob* output_num_blob = this->Output(2);

    // reshape output num blob
    std::vector<TIndex> output_num_shape;
    output_num_shape.push_back(params->cartesian_output.num_size);
    output_num_blob->Reshape(output_num_shape);
    params->cartesian_output.nums = output_num_blob->as<N_DType>();

    // reshape output id & value blob
    TIndex output_length = 0;
    for (auto i = 0; i < params->cartesian_output.num_size; ++i) {
      // calc candidate size of each result
      N_DType candidates_size = 1;
      for (auto j = 0; j < params->input_items.size(); ++j) {
        if (params->input_items[j].num_size == 1) {  // is common input
          candidates_size *= params->input_items[j].nums[0];
        } else {  // is uncommon input
          candidates_size *= params->input_items[j].nums[i];
        }
      }
      output_length += candidates_size;
      params->cartesian_output.nums[i] = candidates_size;
    }

    std::vector<TIndex> output_shape;
    output_shape.push_back(output_length);
    output_id_blob->Reshape(output_shape);
    output_value_blob->Reshape(output_shape);
    params->cartesian_output.ids = output_id_blob->as<K_DType>();
    params->cartesian_output.values = output_value_blob->as<V_DType>();
  }

  void CheckValid() {
    // check input size
    BLAZE_CONDITION_THROW(this->InputSize() % 3 == 0,
                          "cartesian op input size=",
                          this->InputSize());

    // check data type
    for (auto i = 0; i < this->InputSize(); ++i) {
      Blob* x = this->Input(i % 3);
      Blob* y = this->Input(i);
      BLAZE_CONDITION_THROW(x->data_type() == y->data_type(),
                            "x->data_type()=",
                            x->data_type(),
                            " y->data_type()=",
                            y->data_type());
    }

    // id & value size must be equal
    for (auto i = 0; i < this->InputSize() / 3; ++i) {
      Blob* id_blob = this->Input(i * 3);
      Blob* value_blob = this->Input(i * 3 + 1);
      BLAZE_CONDITION_THROW(id_blob->size() == value_blob->size(),
                            "id_blob->size()=",
                            id_blob->size(),
                            " value_blob->size()=",
                            value_blob->size());
    }
  }
};

}  // namespace blaze
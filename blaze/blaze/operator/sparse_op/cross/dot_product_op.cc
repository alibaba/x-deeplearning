/*
 * \file dot_product_op.cc
 * \brief The dot product operator for dot product combine feature.
 */
#include "blaze/operator/sparse_op/cross/dot_product_op.h"
#include "blaze/math/binary_search.h"

namespace blaze {

template <typename K_DType, typename V_DType, typename N_DType>
void RunDotProduct(DotProductParam<K_DType, V_DType, N_DType>& params, bool need_partial_sort) {
  N_DType input_offset = 0;
  N_DType output_offset = 0;
  for (auto i = 0; i < params.num_size; ++i) {
    N_DType start = input_offset;
    N_DType end = input_offset + params.x_nums[i];

    // Step 1: sort x ids
    if (need_partial_sort) {
      QuickSortByID<K_DType, V_DType>(params.x_ids, params.x_values, start, end - 1);
    }

    // Step 2: binary search
    K_DType result_index[params.y_nums[i]];
    BinarySearch<K_DType>(params.x_ids + input_offset, params.x_nums[i],
                          params.y_ids + input_offset, params.y_nums[i], result_index);

    // Step 3: calc dot product params
    float sum_aa = 0.0f;
    float sum_bb = 0.0f;
    float sum_ab = 0.0f;
    float sum_abs_ab = 0.0f;
    float sum_abab = 0.0f;
    bool success = true;
    for (auto j = 0; j < params.y_nums[i]; ++j) {
      float value1 = params.x_values[input_offset + j];
      float value2 = 1.0f;
      if (result_index[j] == -1) {
        success = false;
        break;
      } else {
        value2 = params.y_values[result_index[j]];
      }
      float aa = value1 * value1;
      float bb = value2 * value2;
      float ab = value1 * value2;
      float abs_ab = ab > 0 ? ab : -1 * ab;
      float abab = ab * ab;

      sum_aa += aa;
      sum_bb += bb;
      sum_ab += ab;
      sum_abs_ab += abs_ab;
      sum_abab += abab;
    }

    // Step 4: write result
    if (success) {
      params.z_ids[output_offset] = kSumAaHashID;
      params.z_ids[output_offset + 1] = kSumAbHashID;
      params.z_ids[output_offset + 2] = kSumBbHashID;
      params.z_ids[output_offset + 3] = kSumAbsAbHashID;
      params.z_ids[output_offset + 4] = kSumAbAbHashID;
      params.z_values[output_offset] = sum_aa;
      params.z_values[output_offset + 1] = sum_ab;
      params.z_values[output_offset + 2] = sum_bb;
      params.z_values[output_offset + 3] = sum_abs_ab;
      params.z_values[output_offset + 4] = sum_abab;
      params.z_nums[i] = 5;
      output_offset += params.z_nums[i];
    } else {
      params.z_nums[i] = 0;
    }
    input_offset += params.y_nums[i];
  }

  // Step 5: reshape result ids & values blob shape
  std::vector<TIndex> shape;
  shape.push_back(static_cast<TIndex>(output_offset));
  params.z_id_blob->Reshape(shape);
  params.z_value_blob->Reshape(shape);
};

template <>
bool DotProductOp<CPUContext>::RunOnDevice() {
  Blob* id = this->Input(0);
  Blob* value = this->Input(1);
  Blob* num = this->Input(2);

  // check the validity of dot product op
  CheckValid();

  ID_TYPE_SWITCH(id->data_type(), K_DType, {
  TYPE_SWITCH(value->data_type(), V_DType, {
  ID_TYPE_SWITCH(num->data_type(), N_DType, {
    DotProductParam<K_DType, V_DType, N_DType> params;
    Setup<K_DType, V_DType, N_DType>(&params);
    RunDotProduct<K_DType, V_DType, N_DType>(params, need_partial_sort_);
  });
  });
  });

  return true;
}

REGISTER_CPU_OPERATOR(DotProduct, DotProductOp<CPUContext>);
// Input: X_id, X_value, X_num, Y_id, Y_value, Y_num.
// Output: Z_id, Z_value, Z_num.
OPERATOR_SCHEMA(DotProduct)
  .NumInputs(6)
  .NumOutputs(3)
  .TypeInferenceFunction([](const OperatorDef& def, const std::vector<DataType>& input_type) {
    ArgumentHelper argument_helper(def);
    std::vector<DataType> ret;
    ret.push_back(input_type[0]);
    ret.push_back(input_type[1]);
    ret.push_back(input_type[2]);
    return ret;
  })
  .SetDoc(R"DOC(
The dot product cross.
  )DOC")
  .Input(0, "X_id", "The ids of feature group X")
  .Input(1, "X_value", "The values of feature group X")
  .Input(2, "X_num", "The id num of each ad in feature group X")
  .Input(3, "Y_id", "The ids of feature group Y")
  .Input(4, "Y_value", "The values of feature group Y")
  .Input(5, "Y_num", "The id num of each ad in feature group Y")
  .Output(0, "Z_id", "The ids of dot product feature group Z")
  .Output(1, "Z_value", "The values of dot product feature group Z")
  .Output(2, "Z_num", "The id num of each ad in feature group Z");

}  // namespace blaze

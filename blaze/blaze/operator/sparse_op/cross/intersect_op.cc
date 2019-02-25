/*
 * \file intersect_op.cc 
 * \brief The intersect operator for feature intersect.
 */
#include "blaze/operator/sparse_op/cross/intersect_op.h"
#include "blaze/operator/sparse_op/cross/cross_common_defines.h"
#include "blaze/math/binary_search.h"

namespace blaze {

template <typename K_DType, typename V_DType, typename N_DType>
void RunIntersect(IntersectParam<K_DType, V_DType, N_DType>& params, bool need_sort) {
  // Step 1: sort user features
  if (need_sort) {
    QuickSortByID<K_DType, V_DType>(params.x_ids, params.x_values, 0, params.x_size - 1);
  }

  // Step 2: binary search ad features in user features
  BinarySearch<K_DType>(params.x_ids, params.x_size,
                                          params.y_ids, params.y_size, params.z_ids);

  // Step 3: varify ad feature num
  N_DType total_fea_num = 0;
  for (auto i = 0; i < params.ad_num; ++i) {
    total_fea_num += params.y_nums[i];
  }
  BLAZE_CONDITION_THROW(total_fea_num == params.y_size,
                        "total_fea_num=",
                        total_fea_num,
                        " params.y_size=",
                        params.y_size);

  // Step 4: arrange output blob
  TIndex total_match_count = 0;
  int offset = 0;
  for (auto i = 0; i < params.ad_num; ++i) {
    int ad_match_count = 0;
    for (auto j = 0; j < params.y_nums[i]; ++j) {
      if (params.z_ids[offset] != -1) {
        ad_match_count++;
        auto index = params.z_ids[offset];
        params.z_ids[total_match_count] = params.x_ids[index];
        params.z_values[total_match_count] = params.x_values[index];
        total_match_count++;
      }
      offset++;
    }
    params.z_nums[i] = ad_match_count;
  }

  // Step 5: reshape result ids & values blob shape
  std::vector<TIndex> shape;
  shape.push_back(total_match_count);
  params.z_id_blob->Reshape(shape);
  params.z_value_blob->Reshape(shape);
};

template <>
bool IntersectOp<CPUContext>::RunOnDevice() {
  Blob* id = this->Input(0);
  Blob* value = this->Input(1);
  Blob* num = this->Input(2);

  // check the validity of intersect op
  CheckValid();

  ID_TYPE_SWITCH(id->data_type(), K_DType, {
  TYPE_SWITCH(value->data_type(), V_DType, {
  ID_TYPE_SWITCH(num->data_type(), N_DType, {
    IntersectParam<K_DType, V_DType, N_DType> params;
    Setup<K_DType, V_DType, N_DType>(&params);
    RunIntersect<K_DType, V_DType, N_DType>(params, need_sort_);
  });
  });
  });

  return true;
}

REGISTER_CPU_OPERATOR(Intersect, IntersectOp<CPUContext>);
// Input: X_id, X_value, X_num, Y_id, Y_value, Y_num.
// Output: Z_id, Z_value, Z_num.
OPERATOR_SCHEMA(Intersect)
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
The intersect cross.
  )DOC")
  .Input(0, "X_id", "The ids of feature group X")
  .Input(1, "X_value", "The values of feature group X")
  .Input(2, "X_num", "The id num of each ad in feature group X")
  .Input(3, "Y_id", "The ids of feature group Y")
  .Input(4, "Y_value", "The values of feature group Y")
  .Input(5, "Y_num", "The id num of each ad in feature group Y")
  .Output(0, "Z_id", "The ids of intersected feature group Z")
  .Output(1, "Z_value", "The values of intersected feature group Z")
  .Output(2, "Z_num", "The id num of each ad in feature group Z");

}  // namespace blaze

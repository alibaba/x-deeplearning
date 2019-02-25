/*
 * \file cartesian_product_op.cc
 * \brief The cartesian product operator for cartesian product combine feature.
 */
#include "blaze/operator/sparse_op/cross/cartesian_product_op.h"

namespace {
const int kMaxHashCodeLen = 20;
}  // namespace

namespace blaze {

template <typename K_DType, typename V_DType, typename N_DType>
void RunCartesianProduct(CartesianProductParam<K_DType, V_DType, N_DType>& params) {
  N_DType output_index = 0;
  for (auto i = 0; i < params.cartesian_output.num_size; ++i) {
    auto candidates_size = params.cartesian_output.nums[i];

    // no cartesian product result, skip current process index
    if (candidates_size == 0) {
      for (auto& input_item : params.input_items) {
        if (input_item.num_size != 1 ) {  // is uncommon input
          input_item.process_start = input_item.process_end;
          input_item.process_end = input_item.process_start + input_item.nums[i];
        }
      }
      continue;
    }

    K_DType candidates_id[candidates_size][params.input_items.size()];
    V_DType candidates_value[candidates_size];
    N_DType candidates_num[candidates_size];
    memset(candidates_num, 0, sizeof(N_DType) * candidates_size);

    // Step 1: generate the base
    N_DType pos = 0, size = 0;

    auto &base_item = params.input_items[pos++];
    if (base_item.num_size == 1) {  // is common input
      base_item.process_start = 0;
      base_item.process_end = base_item.nums[0];
    } else {  // is uncommon input
      base_item.process_start = base_item.process_end;
      base_item.process_end = base_item.process_start + base_item.nums[i];
    }

    for (auto j = base_item.process_start; j < base_item.process_end; ++j) {
      auto sp = candidates_num[size];
      candidates_num[size] += 1;
      candidates_id[size][sp] = base_item.ids[j];
      candidates_value[size++] = base_item.values[j];
    }

    // Step 2: process continue for each result
    for (; pos < params.input_items.size(); ++pos) {
      auto& continue_item = params.input_items[pos];
      auto pre_size = size;

      if (continue_item.num_size == 1) {  // is common input
        continue_item.process_start = 0;
        continue_item.process_end = continue_item.nums[0];
      } else {  // is uncommon input
        continue_item.process_start = continue_item.process_end;
        continue_item.process_end = continue_item.process_start + continue_item.nums[i];
      }

      for (auto j = continue_item.process_start + 1; j < continue_item.process_end; ++j) {
        for (auto k = 0; k < pre_size; ++k) {
          candidates_num[size] = candidates_num[k];
          memcpy(candidates_id[size], candidates_id[k], sizeof(K_DType) * candidates_num[size]);
          auto sp = candidates_num[size]++;
          candidates_id[size][sp] = continue_item.ids[j];
          candidates_value[size++] = candidates_value[k] * continue_item.values[j];
        }
      }

      for (auto k = 0; k < pre_size; ++k) {
        auto sp = candidates_num[k]++;
        candidates_id[k][sp] = continue_item.ids[continue_item.process_start];
        candidates_value[k] *= continue_item.values[continue_item.process_start];
      }
    }

    // Step 3: concat compound key & write result
    size_t max_key_length = (kMaxHashCodeLen + 1) * params.input_items.size();
    char convert[max_key_length];
    for (auto j = 0; j < candidates_size; ++j) {
      auto len = snprintf(convert, sizeof(convert), "%lld", candidates_id[j][0]);
      for (auto k = 1; k < params.input_items.size(); ++k) {
        convert[len++] = '+';
        auto strlen = snprintf(convert + len, sizeof(convert) - len, "%lld", candidates_id[j][k]);
        len += strlen;
      }
      LOG_DEBUG("concat result = %s", convert);
      auto hash_code = blaze::MurmurHash64A(convert, len);

      // write output blob
      params.cartesian_output.ids[output_index + j] = static_cast<K_DType>(hash_code);
      params.cartesian_output.values[output_index + j] = candidates_value[j];
      LOG_DEBUG("output_id = %lld", params.cartesian_output.ids[output_index + j]);
      LOG_DEBUG("output_value = %.4f", params.cartesian_output.values[output_index + j]);
    }
    output_index += candidates_size;
  }
}

template <>
bool CartesianProductOp<CPUContext>::RunOnDevice() {
  Blob* id = this->Input(0);
  Blob* value = this->Input(1);
  Blob* num = this->Input(2);

  // check the validity of cartesian op
  CheckValid();

  ID_TYPE_SWITCH(id->data_type(), K_DType, {
  TYPE_SWITCH(value->data_type(), V_DType, {
  ID_TYPE_SWITCH(num->data_type(), N_DType, {
    CartesianProductParam<K_DType, V_DType, N_DType> params;
    Setup<K_DType, V_DType, N_DType>(&params);
    RunCartesianProduct<K_DType, V_DType, N_DType>(params);
  });  // ID_TYPE_SWITCH(num->data_type(), N_DType,
  });  // TYPE_SWITCH(value->data_type(), V_DType,
  });  // ID_TYPE_SWITCH(id->data_type(), K_DType,

  return true;
}

REGISTER_CPU_OPERATOR(CartesianProduct, CartesianProductOp<CPUContext>);
// Input: input1_id, input1_value, input1_num ...
// Output: output_id, output_value, output_index.
OPERATOR_SCHEMA(CartesianProduct)
  .NumInputs(6, 12)
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
The cartesian product cross.
  )DOC");

}  // namespace blaze
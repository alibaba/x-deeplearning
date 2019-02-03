/*
 * \file mock_sparse_puller.cc 
 * \brief The mock sparse puller implementation
 */
#include "blaze/store/mock_sparse_puller.h"

#include "blaze/math/float16.h"
#include "blaze/store/defines.h"

namespace {
float kDefaultWeight = 1.0f;
}  // namespace

namespace blaze {
namespace store {

SparsePuller* CreateSparsePuller() {
  return new MockSparsePuller();
}

Status MockSparsePuller::Get(const std::vector<SparsePullerInput>& input,
                             std::vector<SparsePullerOutput>& output) {
  if (input.size() != output.size()) return kFail;
  for (size_t i = 0; i < input.size(); ++i) {
    const auto& sparse_puller_input = input[i];
    auto& sparse_puller_output = output[i];
    auto status = Get(sparse_puller_input, sparse_puller_output); 
    if (status != kOK) return status;
  }
  return kOK;
}

template <typename ValueType>
static void InitOutputMemory(int dim,
                             const std::vector<SparsePullerInput::Param>& param,
                             std::vector<ValueType*>& out) {
  for (size_t i = 0; i < out.size(); ++i) {
    auto current_param = param[i];
    auto current_out = out[i];
    switch (current_param.udf_type) {
      case kSum:
      case kAvg:
        {
          memset(current_out, 0, sizeof(ValueType) * dim);
        }
        break;
      case kAssign:
        {
          memset(current_out, 0, dim * current_param.trunc_num * sizeof(ValueType));
        }
        break;
    }
  }
}

template <typename KeyType, typename ValueType>
static void ProcessElement(KeyType key,
                           ValueType value,
                           int dim,
                           const std::vector<SparsePullerInput::Param>& param,
                           std::vector<ValueType*>& out,
                           size_t index,
                           size_t size) {
  for (size_t i = 0; i < out.size(); ++i) {
    auto current_param = param[i];
    auto current_out = out[i];
    switch (current_param.udf_type) {
      case kSum:
      case kAvg:
        {
          for (int k = 0; k < dim; ++k) {
            *(current_out) = *(current_out) + (value) * (ValueType)kDefaultWeight;
            ++current_out;
          }
        }
        break;
      case kAssign:
        {
          if (current_param.trunc_direction == kOrder && index >= current_param.trunc_num) continue;
          if (current_param.trunc_direction == kReverse && (size - index) > current_param.trunc_num) continue;

          if (current_param.trunc_direction == kOrder) {
            current_out += dim * index;
          } else {
            current_out += dim * (current_param.trunc_num - (size - index));
          }
          for (int k = 0; k < dim; ++k) {
            *(current_out++) = (value) * (ValueType)kDefaultWeight;
          }
        }
        break;
    }
  }
}

Status MockSparsePuller::Get(const SparsePullerInput& input, SparsePullerOutput& output) {
  if (input.in_item.size() != output.out_item.size()) return kFail;

  EMBEDDING_KEY_TYPE_SWITCH(input.key_type, KeyType, {
    EMBEDDING_VALUE_TYPE_SWITCH(input.value_type, ValueType, {
      EMBEDDING_NUM_TYPE_SWITCH(input.num_type, NumType, {
        KeyType * key = reinterpret_cast<KeyType *>(input.key);
        ValueType* value = reinterpret_cast<ValueType*>(input.value);
        NumType* key_num = reinterpret_cast<NumType*>(input.key_num);
        std::vector<ValueType*> out(output.out_item.size());
        for (size_t i = 0; i < output.out_item.size(); ++i) {
          out[i] = reinterpret_cast<ValueType *>(output.out_item[i].out);
        }
        // enumeration on all batch
        for (size_t i = 0; i < input.key_num_size; ++i) {
          InitOutputMemory(input.dim, input.in_item, out);
          size_t num = static_cast<size_t>(key_num[i]);
          // enumeration on one batch id.
          for (size_t j = 0; j < num; ++j) {
            // process each elements.
            ProcessElement(key[j], value[j], input.dim, input.in_item, out, j, num);
          }
          for (size_t j = 0; j < output.out_item.size(); ++j) {
            if (input.in_item[j].udf_type == kAvg) {
              for (int k = 0; k < input.dim; ++k) {
                out[j][k] = out[j][k] * ValueType(1.0 / num);
              }
            }
            out[j] += output.out_item[j].stride;
          }
          key += num;
          value += num;
        }
      });
    });
  });
  return kOK;
}

}  // namespace store
}  // namespace blaze

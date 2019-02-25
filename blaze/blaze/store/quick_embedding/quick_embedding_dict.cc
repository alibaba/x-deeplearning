/*!
 * \file quick_embedding_dict.cc
 * \desc Sparse parameter dict
 */

#include "blaze/store/quick_embedding/quick_embedding_dict.h"
#include "blaze/store/quick_embedding/udf_processor.h"
#include "blaze/store/defines.h"

#include <fstream>

namespace blaze {
namespace store {

bool QuickEmbeddingDict::SelfCheck(const std::string &url) {
  std::ifstream is(url, std::ios::binary);
  if (!is.is_open()) {
    LOG_ERROR("load quick embedding dict failed! url: %s", url.c_str());
    return false;
  }
  if (!verifier_.Load(&is)) {
    is.close();
    return false;
  }
  is.close();
  return true;
}

Status QuickEmbeddingDict::Load(const std::string &url) {
  std::ifstream is(url, std::ios::binary);
  if (!is.is_open()) {
    LOG_ERROR("load quick embedding dict failed! url: %s", url.c_str());
    return kFail;
  }
  if (!verifier_.Load(&is)) {
    LOG_ERROR("load verifier failed! url: %s", url.c_str());
    is.close();
    return kFail;
  }
  if (verifier_.value_type() == DictValueType::unknown) {
    LOG_ERROR("unknown value type of dict url: %s version: %d", url.c_str(), verifier_.version());
    is.close();
    return kFail;
  }
  if (!trie_.Load(&is)) {
    LOG_ERROR("load trie failed! url: %s", url.c_str());
    is.close();
    return kFail;
  }
  uint16_t gid;
  while (is.read((char *) &gid, sizeof(gid))) {
    if (!hashtables_[gid].Load(&is)) {
      LOG_ERROR("load hashtable failed! url: %s gid: %d", url.c_str(), gid);
      is.close();
      return kFail;
    }
    if (weight_blobs_[gid] == nullptr) {
      SWITCHTYPE_DictValueType(verifier_.value_type(), Type, {
        weight_blobs_[gid].reset(new WeightBlob<Type>());
      })
      if (weight_blobs_[gid] == nullptr) {
        is.close();
        return kFail;
      }
    }
    if (!weight_blobs_[gid]->Load(&is)) {
      LOG_ERROR("load weight blob failed! url: %s gid: %d", url.c_str(), gid);
      is.close();
      return kFail;
    }
  }
  is.close();
  return kOK;
}

Status QuickEmbeddingDict::Get(const std::vector<SparsePullerInput> &input,
                               std::vector<SparsePullerOutput> &output) {
  if (input.size() != output.size()) return kFail;

  for (auto i = 0; i < input.size(); ++i) {
    const auto& sparse_puller_input = input[i];
    auto& sparse_puller_output = output[i];
    Status status = Get(sparse_puller_input, sparse_puller_output);
    if (status != kOK) return status;
  }
  return kOK;
}

Status QuickEmbeddingDict::Get(const SparsePullerInput& input,
                               SparsePullerOutput& output) {
  if (input.in_item.size() != output.out_item.size()) return kFail;

  // table name miss
  uint16_t gid = 0;
  if (!GetGid(input.name, &gid)) return kOK;

  EMBEDDING_KEY_TYPE_SWITCH(input.key_type, KeyType, {
    EMBEDDING_VALUE_TYPE_SWITCH(input.value_type, ValueType, {
      EMBEDDING_VALUE_TYPE_SWITCH(GetBlazeDataType(verifier_.value_type()), DictValueType, {
        EMBEDDING_NUM_TYPE_SWITCH(input.num_type, NumType, {
          KeyType * key = reinterpret_cast<KeyType *>(input.key);
          ValueType* value = reinterpret_cast<ValueType*>(input.value);
          NumType* key_num = reinterpret_cast<NumType*>(input.key_num);
          std::vector<ValueType*> out(output.out_item.size());
          for (size_t i = 0; i < output.out_item.size(); ++i) {
            out[i] = reinterpret_cast<ValueType *>(output.out_item[i].out);
          }
          DictValueType* weights = nullptr;

          // enumeration on all batch
          for (size_t i = 0; i < input.key_num_size; ++i) {
            size_t num = static_cast<size_t>(key_num[i]);
            // init process
            for (size_t k = 0; k < input.in_item.size(); ++k) {
              auto processor = UdfProcessorFactory<DictValueType, ValueType>::Create(input.in_item[k].udf_type);
              processor->InitProcess(input.dim, input.in_item[k], out[k]);
            }

            // element process
            for (size_t j = 0; j < num; ++j) {
              if (!Lookup<DictValueType>(gid, key[j], &weights)) {
                continue;
              }
              for (size_t k = 0; k < input.in_item.size(); ++k) {
                auto processor = UdfProcessorFactory<DictValueType, ValueType>::Create(input.in_item[k].udf_type);
                processor->ElementProcess(value[j], weights, input.dim, j, num, input.in_item[k], out[k]);
              }
            }

            // reduce process
            for (size_t k = 0; k < input.in_item.size(); ++k) {
              auto processor = UdfProcessorFactory<DictValueType, ValueType>::Create(input.in_item[k].udf_type);
              processor->ReduceProcess(input.dim, num, input.in_item[k], out[k]);

              // stride address
              out[k] += output.out_item[k].stride;
            }

            key += num;
            value += num;
          }
        });
      });
    });
  });
  return kOK;
}


QuickEmbeddingDict* CreateQedSparsePuller () {
  return new QuickEmbeddingDict();
}
REGISTER_SPARSE_PULLER_CREATION("qed_sparse_puller", CreateQedSparsePuller);

}  // namespace store
}  // namespace blaze

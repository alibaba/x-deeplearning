/*!
 * \file quick_embedding_dict.h
 * \desc Sparse parameter dict
 */
#pragma once

#include <string>
#include <array>
#include <memory>

#include "blaze/store/sparse_puller.h"
#include "blaze/store/quick_embedding/version_verifier.h"
#include "blaze/store/quick_embedding/trie.h"
#include "blaze/store/quick_embedding/hashtable.h"
#include "blaze/store/quick_embedding/weight_blob.h"

namespace {
const uint32_t kMaxGidSize = 1024;
}  // namespace

namespace blaze {
namespace store {

class QuickEmbeddingDict : public SparsePuller {
 public:
  QuickEmbeddingDict() {}

  ~QuickEmbeddingDict() override {}

  // load quick embedding, kOK if success
  Status Load(const std::string& url) override;

  // pull embedding data, kOK if success
  Status Get(const std::vector<SparsePullerInput>& input,
             std::vector<SparsePullerOutput>& output) override;
  // model self checking, return true if verification is ok
  bool SelfCheck(const std::string &url);

 protected:
  // pull embedding data of single route, kOK if success
  Status Get(const SparsePullerInput& input, SparsePullerOutput& output);

  // get gid by route table name, return true if success
  bool GetGid(const std::string &table_name, uint16_t *gid) const {
    return GetGid(table_name.c_str(), gid);
  }

  // get gid by route table name, return true if success
  bool GetGid(const char *table_name, uint16_t *gid) const {
    return trie_.Lookup(table_name, gid);
  }

  // loop up weights by gid & fid
  template<typename T>
  bool Lookup(uint16_t gid, uint64_t fid, T **dict_value) const {
    if (gid >= kMaxGidSize
        || DictValueTypeFromType<T>::valueType() != verifier_.value_type()) {
      return false;
    }
    uint64_t offset;
    bool flag = hashtables_[gid].Lookup(fid, &offset);
    if (!flag) {
      return false;
    }
    auto weight_blob = static_cast<WeightBlob <T> *>(weight_blobs_[gid].get());
    if (weight_blob == nullptr) {
      return false;
    }
    *dict_value = weight_blob->GetWeights(offset);
    return true;
  }

  // get blaze data type from dict value type
  DataType GetBlazeDataType(DictValueType type) {
    switch (type) {
      case DictValueType::fp32:
        return DataType::kFloat;
      case DictValueType::fp16:
        return DataType::kFloat16;
      case DictValueType::int8:
        return DataType::kInt8;
      default:
        BLAZE_THROW("invalid dict value data type");
    }
  }

 protected:
  VersionVerifier verifier_;
  Trie trie_;
  HashTable hashtables_[kMaxGidSize];
  std::array<std::unique_ptr<Serializable>, kMaxGidSize> weight_blobs_;
};

}  // namespace store
}  // namespace blaze


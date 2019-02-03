/*!
 * \file embedding_builder.h
 * \desc embedding builder, convert offline raw data to embedding bin
 */
#pragma once

#include <cstdint>
#include <string>
#include <fstream>

#include "blaze/store/quick_embedding/version_verifier.h"
#include "blaze/store/quick_embedding/trie.h"
#include "blaze/store/quick_embedding/hashtable.h"
#include "blaze/store/quick_embedding/weight_blob.h"

namespace blaze {
namespace store {

template<typename T>
class EmbeddingBuilder {
 public:
  EmbeddingBuilder() : verifier_(DictValueTypeFromType<T>::valueType()) {
  }

  virtual ~EmbeddingBuilder() = default;

  // schedule multi-thread build
  void Build(const std::string &path,
             const std::string &meta,
             const std::string &output_file,
             int build_thread_num);

 protected:
  // impl of build a file, return true if success
  static bool DoBuild(const std::string &filename,
                      std::uint16_t gid,
                      BulkLoadHashTable* hashtable,
                      WeightBlob<T>* weight_blob);

  // statistics line count of file, return line count
  static int StatisticsLineCount(const std::string &filename);

  // statistics dim size of file, return dim size
  static int StatisticsDimSize(const std::string &filename);

  // parse feature group infos from meta, return fg list
  std::vector<std::string> ParseMeta(const std::string &meta);

  // save to file, return true if success
  bool SaveToFile(const std::string &output_file);

 private:
  VersionVerifier verifier_;
  BulkLoadTrie trie_;
  std::vector<BulkLoadHashTable> hashtable_;
  std::vector<WeightBlob<T> > weight_blob_;
};

}  // namespace store
}  // namespace blaze
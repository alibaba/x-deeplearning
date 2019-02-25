/*!
 * \file embedding_builder.cc
 * \desc embedding builder, convert offline raw data to embedding bin
 */

#include "blaze/store/quick_embedding/embedding_builder.h"
#include "blaze/common/string_util.h"
#include "blaze/common/thread_pool.h"

namespace {
const char kDelimiter = ',';
const int kKeyIdx = 0;
const int kWeightStartIdx = 1;
}  // namespace

namespace blaze {
namespace store {

template<typename T>
void EmbeddingBuilder<T>::Build(const std::string &path,
                                const std::string &meta,
                                const std::string &output_file,
                                int thread_num) {
  auto meta_path = path + '/' + meta;
  auto fg_list = this->ParseMeta(meta_path);
  BLAZE_CONDITION_THROW(!fg_list.empty(), "empty fg list!");

  hashtable_.resize(fg_list.size());
  weight_blob_.resize(fg_list.size());
  std::vector<std::future<bool> > results;

  // start thread pool
  ThreadExecutor thread_executor(thread_num);
  for (auto i = 0; i < fg_list.size(); ++i) {
    const std::string &fg_name = fg_list[i];
    auto filepath = path + '/' + fg_name;
    std::uint16_t gid = static_cast<std::uint16_t>(i);
    trie_.PreInsert(fg_name, i);
    // commit task to thread pool
    results.emplace_back(thread_executor.commit(DoBuild, filepath, gid,
                                                &hashtable_[i], &weight_blob_[i]));
  }
  // wait to shutdown
  thread_executor.shutdown();

  // check result
  for (auto &&result : results) {
    BLAZE_CONDITION_THROW(result.get(), "build failed!");
  }

  // save file
  BLAZE_CONDITION_THROW(SaveToFile(output_file), "export binary file failed!");
}

template<typename T>
std::vector<std::string> EmbeddingBuilder<T>::ParseMeta(const std::string &meta) {
  std::vector<std::string> fg_list;
  std::ifstream fin(meta, std::ios::in);
  if (!fin.is_open()) {
    LOG_ERROR("can not open meta file: %s", meta.c_str());
    return fg_list;
  }

  std::string buff;
  getline(fin, buff);
  while (!fin.eof()) {
    if (!buff.empty()) {
      fg_list.emplace_back(buff);
    }
    getline(fin, buff);
  }
  fin.close();
  return fg_list;
}

template<typename T>
bool EmbeddingBuilder<T>::DoBuild(const std::string &filename,
                                  std::uint16_t gid,
                                  BulkLoadHashTable* hashtable,
                                  WeightBlob<T>* weight_blob) {
  int line_count = StatisticsLineCount(filename);
  if (line_count < 0) {
    return false;
  }
  int dim = StatisticsDimSize(filename);
  if (dim < 0) {
    return false;
  }
  if (!weight_blob->AllocateMemory(line_count * dim)) {
    LOG_ERROR("bad allocate memory of weight blob, filename: %s", filename.c_str());
    return false;
  }

  std::ifstream fin(filename, std::ios::in);
  if (!fin.is_open()) {
    LOG_ERROR("can not open file: %s", filename.c_str());
    return false;
  }

  std::string buff;
  getline(fin, buff);
  while (fin) {
    HashTable::Key key = 0;
    float weights[dim];
    std::vector<std::string> fields = Split(buff, kDelimiter);
    if (kWeightStartIdx + dim == fields.size()) {
      try {
        for (auto i = 0; i < fields.size(); ++i) {
          if (i == kKeyIdx) {
            key = std::stoll(fields[i]);
          } else if (i >= kWeightStartIdx) {
            weights[i - kWeightStartIdx] = stof(fields[i]);
          }
        }

        // insert weight blob
        T *dict_weights = nullptr;
        std::uint64_t offset = weight_blob->InsertWeights(dim, &dict_weights);
        if (nullptr == dict_weights) {
          LOG_ERROR("weight blob out or memory!");
          return false;
        }

        hashtable->PreInsert(key, offset);

        for (int i = 0; i < dim; ++i) {
          dict_weights[i] = weights[i];
        }
      } catch (std::exception& e) {
        LOG_ERROR("lexical cast failed! %s", buff.c_str());
      }
    } else {
      LOG_ERROR("string buff fields num error! %s", buff.c_str());
    }
    getline(fin, buff);
  }
  fin.close();

  // bulk load hashtable
  if (!hashtable->BulkLoad()) {
    LOG_ERROR("bulk load hashtable failed! gid: %hd", gid);
    return false;
  }

  LOG_INFO("task complete, name: %s", filename.c_str());
  return true;
}

template<typename T>
int EmbeddingBuilder<T>::StatisticsLineCount(const std::string &filename) {
  std::ifstream fin(filename, std::ios::in);
  if (!fin.is_open()) {
    LOG_ERROR("can not open file: %s", filename.c_str());
    return -1;
  }

  int line_count = 0;
  std::string buff;
  getline(fin, buff);
  while (fin) {
    line_count++;
    getline(fin, buff);
  }
  fin.close();
  return line_count;
}

template<typename T>
int EmbeddingBuilder<T>::StatisticsDimSize(const std::string &filename) {
  std::ifstream fin(filename, std::ios::in);
  if (!fin.is_open()) {
    LOG_ERROR("can not open file: %s", filename.c_str());
    return -1;
  }
  std::string buff;
  getline(fin, buff);
  if (!fin) {
    LOG_ERROR("can not read first line, file: %s", filename.c_str());
    fin.close();
    return -1;
  }
  fin.close();

  std::vector<std::string> fields = Split(buff, kDelimiter);
  int dim = static_cast<int>(fields.size() - kWeightStartIdx);
  if (dim < 0) {
    LOG_ERROR("invalid field size = ", fields.size());
    return -1;
  }
  return dim;
}

template<typename T>
bool EmbeddingBuilder<T>::SaveToFile(const std::string &output_file) {
  // bulk load trie
  if (!trie_.BulkLoad()) {
    LOG_ERROR("bulk load trie failed!");
    return false;
  }
  LOG_INFO("trie byte size = %ld", trie_.ByteArraySize());

  std::ofstream os(output_file, std::ios::binary);
  if (!os.is_open()) {
    LOG_ERROR("can not open file: %s", output_file.c_str());
    return false;
  }
  // dump version verifier
  if (!verifier_.Dump(&os)) {
    LOG_ERROR("dump version failed!");
    os.close();
    return false;
  }
  // dump trie
  if (!trie_.Dump(&os)) {
    LOG_ERROR("dump trie failed!");
    os.close();
    return false;
  }
  if (hashtable_.size() != weight_blob_.size()) {
    LOG_ERROR("vector size not equal!, hashtable: %d, weight blob: %d",
              hashtable_.size(), weight_blob_.size());
    return false;
  }
  for (auto i = 0; i < hashtable_.size(); ++i) {
    // write gid
    std::uint16_t gid = static_cast<std::uint16_t>(i);
    os.write((char *) &gid, sizeof(gid));
    if (!os.good()) {
      os.close();
      return false;
    }

    // dump hashtable
    if (!hashtable_[i].Dump(&os)) {
      LOG_ERROR("dump hashtable failed! gid: %hd", gid);
      os.close();
      return false;
    }
    LOG_INFO("hashtable byte size = %ld", hashtable_[i].ByteArraySize());

    // dump weights block
    if (!weight_blob_[i].Dump(&os)) {
      LOG_ERROR("dump weights block failed! gid: %hd", gid);
      os.close();
      return false;
    }
    LOG_INFO("weight blob byte size = %ld", weight_blob_[i].ByteArraySize());
  }

  os.close();
  return true;
}

template class EmbeddingBuilder<float>;
template class EmbeddingBuilder<float16>;

}  // namespace store
}  // namespace blaze
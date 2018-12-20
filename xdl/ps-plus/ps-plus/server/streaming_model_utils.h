/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PS_PLUS_SERVER_STREAMING_MODEL_UTILS_H_
#define PS_PLUS_SERVER_STREAMING_MODEL_UTILS_H_

#include "ps-plus/common/tensor.h"
#include "ps-plus/common/status.h"

#include <unordered_set>
#include <unordered_map>
#include <mutex>

namespace ps {
namespace server {

class StreamingModelUtils {
 public:
  struct DenseLog {
    bool clear;
    DenseLog() : clear(true) {}
    void Clear() { clear = true; }
    void Combine(const DenseLog& log) { clear &= log.clear; }
    bool IsClear() { return clear; }
  };

  struct SparseLog {
    std::unordered_set<size_t> write_ids;
    void Clear() {
      write_ids.clear();
    }
    void Combine(const SparseLog& log) {
      write_ids.insert(log.write_ids.begin(), log.write_ids.end());
    }
    bool IsClear() {
      return write_ids.empty();
    }
  };

  struct HashLog {
    struct PairHash { 
      size_t operator()(std::pair<int64_t, int64_t> key) const { 
        int64_t& x = key.first;
        int64_t& y = key.second;
        x = ((x & 0xAAAAAAAAAAAAAAAAL) >> 1) + ((x & 0x5555555555555555L) << 1);
        y = ((y & 0xFFFFFFFF00000000L) >> 32) + ((y & 0x00000000FFFFFFFFL) << 32);
        return x ^ y;
      }
    };
    struct PairEqual {
      bool operator() (std::pair<int64_t, int64_t> lhs, std::pair<int64_t, int64_t> rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
      }
    };
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash, PairEqual> write_ids;
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash, PairEqual> del_ids;
    void Clear() {
      write_ids.clear();
      del_ids.clear();
    }
    void Combine(const HashLog& log) {
      write_ids.insert(log.write_ids.begin(), log.write_ids.end());
      del_ids.insert(log.del_ids.begin(), log.del_ids.end());
    }
    bool IsClear() {
      return write_ids.empty() && del_ids.empty();
    }
  };

  static Status WriteDense(const std::string& var);
  static Status WriteSparse(const std::string& var, const Tensor& data);
  static Status WriteHash(const std::string& var, const Tensor& data);
  static Status DelHash(const std::string& var, const std::vector<int64_t>& data);
  static Status GetDense(std::unordered_map<std::string, DenseLog>* var);
  static Status GetSparse(std::unordered_map<std::string, SparseLog>* var);
  static Status GetHash(std::unordered_map<std::string, HashLog>* var);
 private:
  struct Logger {
    std::mutex mu;
    std::unordered_map<std::string, DenseLog> dense;
    std::unordered_map<std::string, SparseLog> sparse;
    std::unordered_map<std::string, HashLog> hash;

    std::unordered_map<std::string, DenseLog> dense_back;
    std::unordered_map<std::string, SparseLog> sparse_back;
    std::unordered_map<std::string, HashLog> hash_back;
  };
  struct LoggerRegister {
    LoggerRegister();
    ~LoggerRegister();
    std::unique_ptr<Logger> logger;
  };
  static std::mutex mu_;
  static std::unordered_set<Logger*> loggers_;
  static thread_local LoggerRegister logger_;
};

}
}

#endif


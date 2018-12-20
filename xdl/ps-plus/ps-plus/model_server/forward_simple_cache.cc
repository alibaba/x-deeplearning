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

#include "ps-plus/model_server/forward.h"
#include "ps-plus/common/initializer/none_initializer.h"

namespace ps {
namespace modelserver {

class ForwardSimpleCache : public ForwardCache {
 public:
  Status Init(ForwardRun forward, const std::unordered_map<std::string, std::string>& map) override {
    forward_ = forward;
    cache_.inited = false;
    uniq_.reset(new Uniq);
    window_size_ = 0; /* Need to init window_size */
    return Status::Ok();
  }
  void Calc(Tensor ids, Callback cb) override {
    Uniq* uniq = nullptr;
    {
      std::unique_lock<std::mutex> lock(mu_);
      ReadCache(ids, cb);
      if (uniq_->rst.size() >= window_size_) {
        uniq = uniq_.release();
      }
    }
    if (uniq != nullptr) {
      Process(uniq);
    }
  }
  Status Flush() override {
    std::unique_lock<std::mutex> lock(mu_);
    timestamp_ += limit_ + 1;
    return Status::Ok();
  }
 private:
  struct CacheNode {
    char* buffer;
    int64_t timestamp;
  };

  struct Cache {
    bool inited;
    std::vector<std::unique_ptr<char[]>> buffer_internal;
    std::deque<std::pair<int64_t, char*>> buffer_list;
    std::unordered_map<int64_t, CacheNode> cache;
    std::vector<size_t> dims;
    DataType type;
    int64_t block;
  };

  struct UniqNode {
    int64_t req_id;
    int64_t req_offset;
    int64_t uniq_id;
  };

  struct Uniq {
    std::vector<int64_t> uniq;
    std::unordered_map<int64_t, int64_t> uniq_map;
    std::vector<UniqNode> rst_node;
    std::vector<Tensor> rst;
    std::vector<int64_t> rst_size;
    std::vector<Callback> cb;
  };

  void ReadCache(Tensor ids, Callback cb) {
    int req_id = uniq_->rst.size();
    uniq_->rst.emplace_back();
    uniq_->rst_size.push_back(ids.Shape().NumElements());
    uniq_->cb.push_back(cb);
    Tensor& rst = uniq_->rst.back();
    for (int i = 0; i < ids.Shape().NumElements(); i++) {
      int64_t id = ids.Raw<int64_t>()[i];
      auto iter = cache_.cache.find(id);
      if (iter != cache_.cache.end() && iter->second.timestamp > timestamp_ - limit_) {
        if (!rst.Initialized()) {
          cache_.dims[0] = ids.Shape().NumElements();
          rst = Tensor(cache_.type, TensorShape(cache_.dims), new initializer::NoneInitializer);
        }
        memcpy(rst.Raw<char>() + cache_.block * i, iter->second.buffer, cache_.block);
      } else {
        auto uniq_iter = uniq_->uniq_map.find(id);
        int64_t uniq_id;
        if (uniq_iter != uniq_->uniq_map.end()) {
          uniq_id = uniq_iter->second;
        } else {
          uniq_id = uniq_->uniq.size();
          uniq_->uniq.push_back(id);
          uniq_->uniq_map[id] = uniq_id;
        }
        uniq_->rst_node.push_back(UniqNode{.req_id = req_id, .req_offset = i, .uniq_id = uniq_id});
      }
    }
  };

  Status AddCache(const std::vector<int64_t>& ids, Tensor rst) {
    std::unique_lock<std::mutex> lock(mu_);
    std::vector<size_t> dims = rst.Shape().Dims();
    if (dims.empty()) {
      return Status::ArgumentError("Result should not be scalar");
    }
    size_t block = dims[0] == 0 ? 0 : rst.Shape().NumElements() / dims[0] * SizeOfType(rst.Type());
    if (cache_.inited) {
      if (dims.size() != cache_.dims.size()) {
        return Status::ArgumentError("Result dim mismatch to cache");
      }
      for (size_t i = 1; i < dims.size(); i++) {
        if (dims[i] != cache_.dims[i]) {
          return Status::ArgumentError("Result dim mismatch to cache " + std::to_string(i));
        }
      }
    } else {
      cache_.inited = true;
      cache_.dims = dims;
      cache_.type = rst.Type();
      cache_.block = block;
      cache_.buffer_internal.emplace_back(new char[block * kBufferAllocation]);
      for (int i = 0; i < kBufferAllocation; i++) {
        cache_.buffer_list.emplace_front(-1, cache_.buffer_internal.back().get() + block * i);
      }
    }
    for (size_t i = 0; i < ids.size(); i++) {
      if (cache_.buffer_list.front().first < timestamp_ - limit_) {
        cache_.buffer_internal.emplace_back(new char[block * kBufferAllocation]);
        for (int j = 0; j < kBufferAllocation; j++) {
          cache_.buffer_list.emplace_front(-1, cache_.buffer_internal.back().get() + block * j);
        }
      }
      char* buf = cache_.buffer_list.front().second;
      cache_.buffer_list.pop_front();
      memcpy(buf, rst.Raw<char>() + i * block, block);
      cache_.buffer_list.emplace_back(timestamp_, buf);
      int64_t id = ids[i];
      cache_.cache[id] = CacheNode{.buffer = buf, .timestamp=timestamp_};
    }
    return Status::Ok();
  }

  void Process(Uniq* uniq) {
    Tensor t(DataType::kInt64, TensorShape({uniq->uniq.size()}), (char*)(void*)&uniq->uniq[0], new initializer::NoneInitializer);
    forward_(t, [uniq, this](Status st, Tensor rst){
      std::unique_ptr<Uniq> deleter(uniq);
      std::vector<size_t> dims = rst.Shape().Dims();
      if (st.IsOk() && dims[0] != uniq->uniq.size()) {
        st = Status::ArgumentError("Result dim0 should be id size");
      }
      if (st.IsOk()) {
        st = AddCache(uniq->uniq, rst);
      }
      if (!st.IsOk()) {
        for (size_t i = 0; i < uniq->cb.size(); i++) {
          uniq->cb[i](st, Tensor());
        }
        return;
      }
      for (size_t i = 0; i < uniq->rst_size.size(); i++) {
        if (!uniq->rst[i].Initialized()) {
          dims[0] = uniq->rst_size[i];
          uniq->rst[i] = Tensor(rst.Type(), TensorShape(dims), new initializer::NoneInitializer);
        }
      }
      size_t block = dims[0] == 0 ? 0 : rst.Shape().NumElements() / dims[0] * SizeOfType(rst.Type());
      char* ptr = rst.Raw<char>();
      for (auto item : uniq->rst_node) {
        memcpy(uniq->rst[item.req_id].Raw<char>() + item.req_offset * block, ptr + item.uniq_id * block, block);
      }
      for (size_t i = 0; i < uniq->cb.size(); i++) {
        uniq->cb[i](Status::Ok(), uniq->rst[i]);
      }
    });
  }

  static constexpr int kBufferAllocation = 1024;
  ForwardRun forward_;
  std::mutex mu_;
  Cache cache_;
  std::unique_ptr<Uniq> uniq_;
  int64_t timestamp_;
  int64_t limit_;
  int64_t window_size_;
};

FORWARD_REGISTER(ForwardSimpleCache, simple_cache);

}
}


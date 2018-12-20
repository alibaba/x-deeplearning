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

class ForwardUniqueCache : public ForwardCache {
 public:
  Status Init(ForwardRun forward, const std::unordered_map<std::string, std::string>& map) override {
    auto iter = map.find("window_size");
    PS_CHECK_BOOL(iter != map.end(),
                  Status::ArgumentError("ForwardUniqueCache should have argument window_size"));
    window_size_ = atoi(iter->second.c_str());
    forward_ = forward;
    cache_.reset(new Cache);
    return Status::Ok();
  }
  void Calc(Tensor ids, Callback cb) override {
    if (ids.Shape().Size() != 1) {
      cb(Status::ArgumentError("ForwardUniqueCache: ids should be rank-1"), ps::Tensor());
      return;
    }
    if (ids.Type() != DataType::kInt64) {
      cb(Status::ArgumentError("ForwardUniqueCache: ids should be int64"), ps::Tensor());
      return;
    }
    Cache* process = nullptr;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cache_->id_map.emplace_back();
      cache_->cb.push_back(cb);
      auto& idmap = cache_->id_map.back();
      for (int i = 0; i < ids.Shape().NumElements(); i++) {
        int64_t id = ids.Raw<int64_t>()[i];
        int64_t uniq_id;
        auto iter = cache_->uniq_map.find(id);
        if (iter == cache_->uniq_map.end()) {
          uniq_id = cache_->tensor.size();
          cache_->uniq_map[id] = uniq_id;
          cache_->tensor.push_back(id);
        } else {
          uniq_id = iter->second;
        }
        idmap.push_back(uniq_id);
      }
      if (cache_->cb.size() == window_size_) {
        process = cache_.release();
        cache_.reset(new Cache);
      }
    }
    if (process != nullptr) {
      Process(process);
    }
  }
  Status Flush() override {
    std::unique_lock<std::mutex> lock(mu_);
    for (auto&& cb : cache_->cb) {
      cb(Status::NetworkError("ForwardUniqueCache: Server is reset"), ps::Tensor());
    }
    cache_.reset(new Cache);
    return Status::Ok();
  }
 private:
  struct Cache {
    std::vector<int64_t> tensor;
    std::unordered_map<int64_t, int64_t> uniq_map;
    std::vector<std::vector<int64_t>> id_map;
    std::vector<Callback> cb;
  };
  void Process(Cache* cache) {
    Tensor t(DataType::kInt64, TensorShape({cache->tensor.size()}), (char*)(void*)&cache->tensor[0], new initializer::NoneInitializer);
    forward_(t, [cache](Status st, Tensor rst){
      std::unique_ptr<Cache> deleter(cache);
      std::vector<size_t> dims;
      if (st.IsOk()) {
        dims = rst.Shape().Dims();
      }
      if (st.IsOk() && dims.empty()) {
        st = Status::ArgumentError("Result should not be scalar");
      }
      if (st.IsOk() && dims[0] != cache->tensor.size()) {
        st = Status::ArgumentError("Result dim0 should be id size");
      }
      if (!st.IsOk()) {
        for (size_t i = 0; i < cache->cb.size(); i++) {
          cache->cb[i](st, Tensor());
        }
        return;
      }
      size_t block = dims[0] == 0 ? 0 : rst.Shape().NumElements() / dims[0] * SizeOfType(rst.Type());
      char* ptr = rst.Raw<char>();
      for (size_t i = 0; i < cache->id_map.size(); i++) {
        dims[0] = cache->id_map[i].size();
        Tensor split_rst(rst.Type(), TensorShape(dims), new initializer::NoneInitializer);
        char* split_ptr = split_rst.Raw<char>();
        for (auto item : cache->id_map[i]) {
          memcpy(split_ptr, ptr + item * block, block);
          split_ptr += block;
        }
        cache->cb[i](Status::Ok(), split_rst);
      }
    });
  }
  ForwardRun forward_;
  int window_size_;
  std::mutex mu_;
  std::unique_ptr<Cache> cache_;
};

FORWARD_REGISTER(ForwardUniqueCache, unique_cache);

}
}


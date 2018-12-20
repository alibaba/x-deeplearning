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

#include "ps-plus/model_server/backward.h"
#include "ps-plus/common/initializer/none_initializer.h"

namespace ps {
namespace modelserver {

class BackwardUniqueCache : public BackwardCache {
 public:
  Status Init(BackwardRun backward, const std::unordered_map<std::string, std::string>& map) override {
    auto iter = map.find("window_size");
    PS_CHECK_BOOL(iter != map.end(),
                  Status::ArgumentError("BackwardUniqueCache should have argument window_size"));
    window_size_ = atoi(iter->second.c_str());
    backward_ = backward;
    cache_.reset(new Cache);
    cache_->inited = false;
    return Status::Ok();
  }
  void Calc(Tensor ids, Tensor grads, Callback cb) override {
    std::vector<size_t> id_dims = ids.Shape().Dims();
    std::vector<size_t> grad_dims = grads.Shape().Dims();
    if (id_dims.size() != 1) {
      cb(Status::ArgumentError("BackwardUniqueCache: ids should be rank-1"));
      return;
    }
    if (ids.Type() != DataType::kInt64) {
      cb(Status::ArgumentError("BackwardUniqueCache: ids should be int64"));
      return;
    }
    if (grad_dims.size() == 0) {
      cb(Status::ArgumentError("BackwardUniqueCache: grads should not be rank-0"));
      return;
    }
    if (grad_dims[0] != id_dims[0]) {
      cb(Status::ArgumentError("BackwardUniqueCache: grads.dim[0] should be equal to ids"));
      return;
    }
    Cache* process = nullptr;
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (cache_->inited) {
        if (grad_dims.size() != cache_->grad_dims.size()) {
          std::string x, y;
          for (auto item : grad_dims) { x += std::to_string(item) + ","; }
          for (auto item : cache_->grad_dims) { y += std::to_string(item) + ","; }
          cb(Status::ArgumentError("BackwardUniqueCache: grad shape mismatch to other worker (" + x + ") vs (" + y + ")"));
          return;
        }
        if (grads.Type() != cache_->grad_type) {
          cb(Status::ArgumentError("BackwardUniqueCache: grad type mismatch to other worker" + std::to_string(grads.Type()) + " vs " + std::to_string(cache_->grad_type)));
          return;
        }
      } else {
        cache_->grad_dims = grad_dims;
        cache_->grad_type = grads.Type();
        cache_->grad_dims[0] = 0;
        cache_->inited = true;
      }
      cache_->cb.push_back(cb);
      size_t block = grad_dims[0] == 0 ? 0 : grads.Shape().NumElements() / grad_dims[0];
      CASES(grads.Type(), {
        for (int i = 0; i < ids.Shape().NumElements(); i++) {
          int64_t id = ids.Raw<int64_t>()[i];
          int64_t uniq_id;
          auto iter = cache_->uniq_map.find(id);
          if (iter == cache_->uniq_map.end()) {
            uniq_id = cache_->id_buffer.size();
            cache_->uniq_map[id] = uniq_id;
            cache_->id_buffer.push_back(id);
            cache_->grad_buffer.resize(cache_->grad_buffer.size() + sizeof(T) * block, 0);
          } else {
            uniq_id = iter->second;
          }
          T* buffer = (T*)(void*)&cache_->grad_buffer[0] + uniq_id * block;
          T* src_buffer = grads.Raw<T>() + i * block;
          for (size_t k = 0; k < block; k++) {
            *buffer++ += *src_buffer++;
          }
        }
      });
      if (cache_->cb.size() == window_size_) {
        process = cache_.release();
        cache_.reset(new Cache);
        cache_->inited = false;
      }
    }
    if (process != nullptr) {
      Process(process);
    }
  }
  Status Flush() override {
    std::unique_lock<std::mutex> lock(mu_);
    for (auto&& cb : cache_->cb) {
      cb(Status::NetworkError("BackwardUniqueCache: Server is reset"));
    }
    cache_.reset(new Cache);
    cache_->inited = false;
    return Status::Ok();
  }
 private:
  struct Cache {
    bool inited;
    std::unordered_map<int64_t, int64_t> uniq_map;
    std::vector<int64_t> id_buffer;
    std::vector<char> grad_buffer;
    DataType grad_type;
    std::vector<size_t> grad_dims;
    std::vector<Callback> cb;
  };
  void Process(Cache* cache) {
    Tensor ids(DataType::kInt64, TensorShape({cache->id_buffer.size()}), (char*)(void*)&cache->id_buffer[0], new initializer::NoneInitializer);
    cache->grad_dims[0] = cache->id_buffer.size();
    Tensor grads(cache->grad_type, TensorShape(cache->grad_dims), &cache->grad_buffer[0], new initializer::NoneInitializer);
    backward_(ids, grads, [cache](Status st){
      for (auto&& item : cache->cb) {
        item(st);
      }
      delete cache;
    });
  }
  BackwardRun backward_;
  int window_size_;
  std::mutex mu_;
  std::unique_ptr<Cache> cache_;
};

BACKWARD_REGISTER(BackwardUniqueCache, unique_cache);

}
}


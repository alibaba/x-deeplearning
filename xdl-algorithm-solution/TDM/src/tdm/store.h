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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#ifndef TDM_STORE_H_
#define TDM_STORE_H_

#include <string>
#include <vector>
#include <unordered_map>

#include "tdm/cache.h"

namespace tdm {

const size_t kBatchSize = 500;

class StoreFactory;
class KVItemList;

class Store {
 public:
  virtual ~Store();

  virtual bool Init(const std::string& config);

  virtual bool Get(const std::string& key, std::string* value) = 0;
  virtual bool Put(const std::string& key, const std::string& value) = 0;

  virtual std::vector<bool>
  MGet(const std::vector<std::string>& keys,
       std::vector<std::string>* values) = 0;

  virtual std::vector<bool> MPut(const std::vector<std::string>& keys,
                                 const std::vector<std::string>& values) = 0;

  virtual bool Remove(const std::string& key) = 0;

  static Store* NewStore(const std::string& config);

  static void DestroyStore(Store* store);

  static void RegisterStoreFactory(const std::string& type,
                                   StoreFactory* factory);

  void LoadData(const std::string& filename);
  virtual bool Dump(const std::string& filename) = 0;

 protected:
  using KVMap = std::unordered_map<std::string, std::string>;
  static KVMap ParseConfig(const std::string& config);

 private:
  static std::unordered_map<std::string, StoreFactory*> factory_;
};

class StoreFactory {
 public:
  virtual ~StoreFactory() {
  }

  virtual Store* NewStore(const std::string& config) = 0;
};

class CachedStore: public Store {
 public:
  CachedStore();
  ~CachedStore();
  virtual void Persist(const std::vector<std::string>& keys);

  bool enable_cache() {
    return enable_cache_;
  }

  void set_enable_cache(bool enable_cache) {
    enable_cache_ = enable_cache;
  }

 protected:
  virtual void Cache(const std::string& key, const std::string& value);
  virtual bool FindInCache(const std::string& key, std::string* value);
  virtual std::vector<bool> FindInCache(const std::vector<std::string>& keys,
                                        std::vector<std::string>* values);

 protected:
  bool enable_cache_;
  LRUCache persist_kv_;
  LRUCache cached_kv_;
};

}  // namespace tdm

#endif  // TDM_STORE_H_

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

#include "tdm/store.h"

#include <stdio.h>

#include "tdm/common.h"
#include "tdm/local_store.h"
#include "tdm/store_kv.pb.h"

namespace tdm {

std::unordered_map<std::string, StoreFactory*> Store::factory_;

Store::~Store() {
}

bool Store::Init(const std::string& config) {
  (void) config;
  return false;
}

std::unordered_map<std::string, std::string>
Store::ParseConfig(const std::string& config) {
  return tdm::ParseConfig(config);
}

Store* Store::NewStore(const std::string& config) {
  auto conf = ParseConfig(config);
  std::string type = "LOCAL";
  auto it = conf.find("type");
  if (it != conf.end()) {
    type = it->second;
  }
  auto fac_it = factory_.find(type);
  if (fac_it == factory_.end()) {
    fprintf(stderr, "No factory for type: %s\n", type.c_str());
    return nullptr;
  }

  auto store = fac_it->second->NewStore(config);
  return store;
}

void Store::LoadData(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    fprintf(stderr, "Can not open file: %s\n", filename.c_str());
    return;
  }

  std::vector<std::string> keys;
  std::vector<std::string> values;
  int num = 0;
  size_t ret = fread(&num, sizeof(num), 1, fp);
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    if (fread(const_cast<char*>(content.data()), 1, num, fp)
        != static_cast<size_t>(num)) {
      fprintf(stderr, "Read from file: %s failed, invalid format.\n",
              filename.c_str());
      break;
    }
    KVItem item;
    if (!item.ParseFromString(content)) {
      fprintf(stderr, "Parse from file: %s failed.\n", filename.c_str());
      break;
    }
    keys.push_back(item.key());
    values.push_back(item.value());
    if (keys.size() >= kBatchSize) {
      MPut(keys, values);
      keys.clear();
      values.clear();
    }
    ret = fread(&num, sizeof(num), 1, fp);
  }

  if (!keys.empty()) {
    MPut(keys, values);
  }
  fclose(fp);
}

void Store::DestroyStore(Store* store) {
  delete store;
}

void Store::RegisterStoreFactory(const std::string& type,
                                 StoreFactory* factory) {
  factory_.insert({type, factory});
}

CachedStore::CachedStore(): enable_cache_(true) {
}

CachedStore::~CachedStore() {
  persist_kv_.Destroy();
  cached_kv_.Destroy();
}

void CachedStore::Persist(const std::vector<std::string>& keys) {
  std::vector<std::string> values(keys.size());
  auto ret = MGet(keys, &values);
  for (size_t i = 0; i < keys.size(); ++i) {
    if (ret[i]) {
      persist_kv_.Put(keys[i], values[i]);
    }
  }
}

void CachedStore::Cache(const std::string& key, const std::string& value) {
  cached_kv_.Put(key, value);
}

bool CachedStore::FindInCache(const std::string& key, std::string* value) {
  auto ret = persist_kv_.Get(key, value);
  if (!ret) {
    ret = cached_kv_.Get(key, value);
  }
  return ret;
}

std::vector<bool>
CachedStore::FindInCache(const std::vector<std::string>& keys,
                         std::vector<std::string>* values) {
  std::vector<bool> ret(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    ret[i] = persist_kv_.Get(keys[i], &values->at(i));
    if (!ret[i]) {
      ret[i] = cached_kv_.Get(keys[i], &values->at(i));
    }
  }
  return ret;
}

}  // namespace tdm

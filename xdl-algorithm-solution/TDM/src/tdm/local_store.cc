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

#include "tdm/local_store.h"

#include <stdio.h>

#include <string>
#include <vector>

#include "tdm/store_kv.pb.h"

namespace tdm {

LocalStore::LocalStore(): Store() {
}

LocalStore::~LocalStore() {
}

bool LocalStore::Init(const std::string& config) {
  (void) config;
  return true;
}

bool LocalStore::Dump(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(), "wb");
  if (fp == nullptr) {
    return false;
  }

  for (auto it = data_.begin(); it != data_.end(); ++it) {
    KVItem item;
    item.set_key(it->first);
    item.set_value(it->second);
    std::string content;
    if (!item.SerializeToString(&content)) {
      fclose(fp);
      return false;
    }
    int len = content.size();
    fwrite(&len, sizeof(len), 1, fp);
    fwrite(content.data(), 1, len, fp);
  }

  fclose(fp);
  return true;
}

bool LocalStore::Get(const std::string& key, std::string* value) {
  auto it = data_.find(key);
  if (it == data_.end()) {
  return false;
  }
  *value = it->second;
  return true;
}

bool LocalStore::Put(const std::string& key, const std::string& value) {
  return data_.insert({key, value}).second;
}

std::vector<bool>
LocalStore::MGet(const std::vector<std::string>& keys,
                 std::vector<std::string>* values) {
  std::vector<bool> ret(keys.size(), false);
  if (values == nullptr) {
    return ret;
  }
  if (values->size() != keys.size()) {
    values->resize(keys.size());
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    ret[i] = Get(keys[i], &values->at(i));
  }
  return ret;
}

std::vector<bool>
LocalStore::MPut(const std::vector<std::string>& keys,
                 const std::vector<std::string>& values) {
  std::vector<bool> rets(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    rets[i] = Put(keys[i], values[i]);
  }
  return rets;
}

bool LocalStore::Remove(const std::string& key) {
  data_.erase(key);
  return true;
}

class LocalStoreFactory: public StoreFactory {
 public:
  Store* NewStore(const std::string& config) {
    LocalStore* store = new LocalStore();
    if (!store->Init(config)) {
      delete store;
      return NULL;
    }
    return store;
  }
};

class LocalStoreRegister {
 public:
  LocalStoreRegister() {
    Store::RegisterStoreFactory("LOCAL", new LocalStoreFactory);
  }
};

static LocalStoreRegister registrar;

}  // namespace tdm

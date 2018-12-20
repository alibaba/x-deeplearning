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

// Copyright 2018 Alibaba Inc. All Rights Conserved.

#include "tdm/cache.h"

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

namespace tdm {

Cache::Cache(): size_in_kbytes_(0), initialized_(false) {
}

Cache::Cache(int nkb): size_in_kbytes_(nkb), initialized_(false) {
}

LRUCache::LRUCache(): Cache() {
}

LRUCache::LRUCache(int nkb): Cache(nkb) {
  Init(nkb);
}

LRUCache::~LRUCache() {
  Destroy();
}

bool LRUCache::Init(int nkb) {
  if (initialized_) {
    return true;
  }
  return false;
}

bool LRUCache::Get(const std::string& key, std::string* value) const {
  if (value == NULL) {
    return false;
  }
  return false;
}

bool LRUCache::Put(const std::string& key, const std::string& value) {
  return false;
}

bool LRUCache::Put(const std::string& key,
                   const std::string& value, int expire) {
  return false;
}

bool LRUCache::Remove(const std::string& key) {
  return false;
}

void LRUCache::Clear() {
}

void LRUCache::Destroy() {
}

}  // namespace tdm

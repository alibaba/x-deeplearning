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

#ifndef TDM_CACHE_H_
#define TDM_CACHE_H_

#include <string>
#include <vector>

namespace tdm {

class Cache {
 public:
  Cache();
  explicit Cache(int nkb);

  virtual ~Cache() {
  }

  virtual bool Init(int nkb) = 0;
  virtual bool Get(const std::string& key,
                   std::string* value) const = 0;
  virtual bool Put(const std::string& key,
                   const std::string& value) = 0;
  virtual bool Put(const std::string& key,
                   const std::string& value, int expire) = 0;
  virtual bool Remove(const std::string& key) = 0;
  virtual void Clear() = 0;
  virtual void Destroy() = 0;

  bool initialized() const {
    return initialized_;
  }

 protected:
  int size_in_kbytes_;
  bool initialized_;
};

class LRUCache: public Cache {
 public:
  LRUCache();
  explicit LRUCache(int nkb);
  virtual ~LRUCache();

  bool Init(int nkb) override;
  bool Get(const std::string& key, std::string* value) const override;
  bool Put(const std::string& key, const std::string& value) override;
  bool Put(const std::string& key,
           const std::string& value, int expire) override;
  bool Remove(const std::string& key) override;
  void Clear() override;
  void Destroy() override;
};

}  // namespace tdm

#endif  // TDM_CACHE_H_

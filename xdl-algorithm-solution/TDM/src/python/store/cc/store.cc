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

#include "include/api.h"

#include "tdm/store.h"

store_handler API(new)(const std::string& config) {
  return tdm::Store::NewStore(config);
}

void API(load)(store_handler handler, const std::string& filename) {
  if (handler != NULL) {
    reinterpret_cast<tdm::Store*>(handler)->LoadData(filename);
  }
}

void API(close)(store_handler handler) {
  return tdm::Store::DestroyStore(reinterpret_cast<tdm::Store*>(handler));
}

int API(put)(store_handler handler,
             const std::string& key,
             const std::string& value) {
  if (handler == nullptr) {
    return 0;
  }

  return reinterpret_cast<tdm::Store*>(handler)->Put(key, value);
}

int API(get)(store_handler handler,
             const std::string& key, std::string* value) {
  if (handler == nullptr) {
    return 0;
  }

  return reinterpret_cast<tdm::Store*>(handler)->Get(key, value);
}

//////////////  Batch operation interface  ///////////////

int API(mget)(store_handler handler,
              const std::vector<std::string>& keys,
              std::vector<std::string>* values) {
  if (handler == nullptr) {
    return 0;
  }
  reinterpret_cast<tdm::Store*>(handler)->MGet(keys, values);
  return 1;
}

int API(mput)(store_handler handler,
              const std::vector<std::string>& keys,
              const std::vector<std::string>& values) {
  if (handler == nullptr) {
    return 0;
  }
  reinterpret_cast<tdm::Store*>(handler)->MPut(keys, values);
  return 1;
}

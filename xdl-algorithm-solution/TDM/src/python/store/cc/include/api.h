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

#ifndef STORE_API_H_
#define STORE_API_H_

#include <string>
#include <vector>

#define API(name) STORE_API_##name

typedef void* store_handler;

store_handler API(new)(const std::string& config);

void API(load)(store_handler handler, const std::string& filename);

void API(close)(store_handler handler);

int API(put)(store_handler handler,
             const std::string& key,
             const std::string& value);

int API(get)(store_handler handler,
             const std::string& key, std::string* value);

//////////////  Batch operation interface  ///////////////

int API(mget)(store_handler handler,
              const std::vector<std::string>& keys,
              std::vector<std::string>* values);

int API(mput)(store_handler handler,
              const std::vector<std::string>& keys,
              const std::vector<std::string>& values);

#endif  // STORE_API_H_

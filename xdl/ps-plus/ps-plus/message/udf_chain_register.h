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

#ifndef PS_MESSAGE_UDF_CHAIN_REGISTER_H_
#define PS_MESSAGE_UDF_CHAIN_REGISTER_H_

#include <vector>
#include <string>
#include <utility>

namespace ps {

struct UdfChainRegister {
  struct UdfDef {
    std::vector<std::pair<int, int>> inputs;
    std::string udf_name;
  };
  size_t hash;
  std::vector<UdfDef> udfs;
  std::vector<std::pair<int, int>> outputs;
};

}

#endif


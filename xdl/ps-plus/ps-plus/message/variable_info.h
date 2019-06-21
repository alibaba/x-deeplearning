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

#ifndef PS_MESSAGE_VARIABLE_INFO_H_
#define PS_MESSAGE_VARIABLE_INFO_H_

#include <vector>
#include <string>
#include <utility>
#include <unordered_map>

#include "ps-plus/common/types.h"

namespace ps {

struct VariableInfo {
  enum Type {
    kIndex,
    kHash
  };
  struct Part {
    size_t server;
    size_t size;
  };

  struct Id {
    int64_t x;
    int64_t y;
  };

  struct FeatureStats {
    unsigned short unseen_times;
    float cvm_show, cvm_clk;
  };

  Type type;
  std::string name;
  std::vector<Part> parts;
  std::vector<int64_t> shape;
  DataType datatype;
  std::unordered_map<std::string, std::string> args;
  size_t visit_time;
  int64_t dense_visit_ids;
  int64_t sparse_visit_ids;
  std::vector<Id> ids;
  std::vector<FeatureStats> stats;
  std::unordered_map<int64_t, size_t> stats_index;
};

struct VariableInfoCollection {
  std::vector<VariableInfo> infos;
};

struct DecayInfoCollection {
  std::unordered_map<std::string, std::vector<int64_t>> decay_infos;
};

}

#endif


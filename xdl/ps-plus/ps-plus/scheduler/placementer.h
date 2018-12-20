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

#ifndef PS_SCHEDULER_PLACEMENTER_H_
#define PS_SCHEDULER_PLACEMENTER_H_

#include "ps-plus/message/variable_info.h"
#include "ps-plus/common/plugin.h"
#include "ps-plus/common/status.h"

namespace ps {
namespace scheduler {

class Placementer {
 public:
  struct Arg {
    size_t net;
    size_t mem;
    size_t query;
  };
  virtual ~Placementer() {}
  virtual Status Placement(const std::vector<VariableInfo>& inputs, std::vector<VariableInfo>* outputs, const Arg& arg, size_t server) = 0;
};

}
}

#endif


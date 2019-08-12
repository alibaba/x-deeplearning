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


#include "xdl/data_io/op/debug_op.h"

#include <iostream>

namespace xdl {
namespace io {

bool DebugOP::Init(const std::map<std::string, std::string> &params) {
  for (auto &kv: params) {
    std::cout << "init: " << kv.first << "->" << kv.second << std::endl;
  }
  return true;
}

bool DebugOP::Run(SampleGroup *sample_group) {
  auto s = sample_group->ShortDebugString();
  std::cout << "sg: " << s << std::endl;
  return true;
}

XDL_REGISTER_IOP(DebugOP)

}  // namespace io
}  // namespace xdl
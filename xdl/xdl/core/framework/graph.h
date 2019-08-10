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

#ifndef XDL_CORE_FRAMEWORK_GRAPH_H_
#define XDL_CORE_FRAMEWORK_GRAPH_H_

#include <string>
#include <vector>

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/device_converter.h"
#include "xdl/core/framework/graph_def.h"

namespace xdl {

struct Node {
  static constexpr int kDependency = -1;
  struct Input {
    int node_id;
    int output_id;
  };
  struct Output {
    int output_id;
    int node_id;
    int input_id;
  };
  std::string name;
  std::string op_name;
  // real input size, do not count dependency
  size_t input_size;
  std::vector<Input> inputs;
  std::vector<Output> outputs;
  std::unique_ptr<OpKernelBase> op;
  OpKernelContextArg arg;
};

struct Graph {
  // Source Node Id
  static constexpr int kSource = 0;
  static constexpr int kSink = 1;
  std::vector<Node> nodes;
  std::map<std::pair<Device*, Device*>, DeviceConverter*> device_converter;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_GRAPH_H_


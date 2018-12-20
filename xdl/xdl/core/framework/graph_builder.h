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

#ifndef XDL_CORE_FRAMEWORK_GRAPH_BUILDER_H_
#define XDL_CORE_FRAMEWORK_GRAPH_BUILDER_H_

#include <unordered_map>
#include <string>

#include "xdl/core/framework/graph_def.h"
#include "xdl/core/framework/graph.h"

namespace xdl {

class GraphBuilder {
 public:
  GraphBuilder(const GraphDef& def, const OutputSpec& output, Graph* graph)
    : def_(def), output_(output), graph_(graph) {}
  Status Build();
 private:
  Status Prepare();
  Status BuildNodes();
  Status BuildNode(const NodeDef& def, Node* node);
  Status BuildSink(Node* node);
  Status AppendOutputs();
  Status AddDeviceConverter();
  Status BuildSource(Node* node);
  Status ParseInput(const std::string& spec, Node::Input* result);
  Status CreateDeviceConverter(Device* src, Device* dst);
  Status CheckDAG();
  Status CheckOutputOverflow();
  GraphDef def_;
  OutputSpec output_;
  Graph* graph_;
  std::unordered_map<std::string, int> node_id_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_GRAPH_BUILDER_H_


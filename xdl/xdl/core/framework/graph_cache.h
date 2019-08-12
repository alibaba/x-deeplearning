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

#ifndef XDL_CORE_FRAMEWORK_GRAPH_CACHE_H_
#define XDL_CORE_FRAMEWORK_GRAPH_CACHE_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/singleton.h"
#include "xdl/core/framework/graph.h"
#include "xdl/core/framework/graph_def.h"

namespace xdl {

class GraphCache : public Singleton<GraphCache> {
 public:
  using GraphCreator =
    std::function<Status(const GraphDef&, const OutputSpec& output, Graph**)>;
  Status GetGraph(GraphCreator creator,
                  const GraphDef& def,
                  const OutputSpec& output,
                  Graph** g);
 private:
  struct GraphToken {
    int64_t graph_hash;
    std::vector<std::string> outputs;
  };
  struct GraphTokenHash {
    size_t operator()(const GraphToken& token) const;
  };
  struct GraphTokenEqual {
    bool operator()(const GraphToken& lhs, const GraphToken& rhs) const;
  };
  std::mutex mu_;
  std::unordered_map<GraphToken, std::unique_ptr<Graph>,
                     GraphTokenHash, GraphTokenEqual> graph_map_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_GRAPH_CACHE_H_


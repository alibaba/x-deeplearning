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

#include "xdl/core/framework/graph_cache.h"

namespace xdl {

Status GraphCache::GetGraph(GraphCreator creator,
                            const GraphDef& def,
                            const OutputSpec& output,
                            Graph** g) {
  std::unique_lock<std::mutex> lock(mu_);
  GraphToken token;
  token.graph_hash = def.hash;
  token.outputs = output.output;
  auto iter = graph_map_.find(token);
  if (iter != graph_map_.end()) {
    *g = iter->second.get();
    return Status::Ok();
  }
  XDL_CHECK_STATUS(creator(def, output, g));
  graph_map_[token].reset(*g);
  return Status::Ok();
}

size_t GraphCache::GraphTokenHash::operator()(const GraphToken& token) const {
  std::hash<std::string> hasher;
  constexpr size_t P = 10240319;
  size_t r = token.graph_hash;
  for (auto&& item : token.outputs) {
    r = r * P + hasher(item);
  }
  return r;
}

bool GraphCache::GraphTokenEqual::operator()(const GraphToken& lhs,
                                             const GraphToken& rhs) const {
  if (lhs.graph_hash != rhs.graph_hash) {
    return false;
  }
  if (lhs.outputs.size() != rhs.outputs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.outputs.size(); i++) {
    if (lhs.outputs[i] != rhs.outputs[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace xdl


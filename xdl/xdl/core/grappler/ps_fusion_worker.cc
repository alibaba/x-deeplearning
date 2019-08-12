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

#include "xdl/core/grappler/ps_fusion_worker.h"
#include <unordered_map>
#include <unordered_set>

namespace xdl {

Status FusionWorker::Init(
    GraphDef* graph, 
    OutputSpec* output) {
  for (auto& item: graph->node) {
    nodes_[item.name] = &item;
  }

  for (auto& name : output->output) {
    output_names_.push_back(name);
    NodeDef* node = nullptr;
    XDL_CHECK_STATUS(GetNodeByName(name, &node));
    output_nodes_.push_back(node);
  }

  graph_ = graph;
  output_ = output;
  return Status::Ok();
}

Status FusionWorker::GetNodeByName(
    const std::string& name,
    NodeDef** node) {
  std::string node_name;
  if (name.size() > 0 && name[0] == '^') {
    node_name = name.substr(1);
  } else {
    size_t pos = name.find(':');
    XDL_CHECK_COND(pos != std::string::npos,
                   Status::ArgumentError("Node Input Error, "
                                         ": not found " + name));
    node_name = name.substr(0, pos);
  }

  auto iter = nodes_.find(node_name);
  XDL_CHECK_COND(iter != nodes_.end(),
                 Status::ArgumentError("not found node:" + name));
  *node = iter->second;
  return Status::Ok();
}

Status FusionWorker::ClusterNodes(
    const std::function<bool(NodeDef*)>& match_func,
    GraphDef* graph, 
    std::vector<std::set<NodeDef*> >* clusters) {
  std::unordered_set<NodeDef*> nodes_set(output_nodes_.begin(), output_nodes_.end());
  std::vector<NodeDef*> bfs(nodes_set.begin(), nodes_set.end());
  std::unordered_map<NodeDef*, std::vector<NodeDef*>> outputs;
  for (size_t p = 0; p < bfs.size(); p++) {
    NodeDef* cur = bfs[p];
    for (auto&& in : cur->input) {
      NodeDef* in_node;
      XDL_CHECK_STATUS(GetNodeByName(in, &in_node));
      if (nodes_set.find(in_node) == nodes_set.end()) {
        nodes_set.insert(in_node);
        bfs.push_back(in_node);
      }
      outputs[in_node].push_back(cur);
    }
  }

  std::vector<NodeDef*> queue;
  std::unordered_map<NodeDef*, size_t> ref;
  std::unordered_map<NodeDef*, size_t> level;
  for (auto item : bfs) {
    if (item->input.empty()) {
      queue.push_back(item);
    }
    ref[item] = item->input.size();
  }
  for (size_t p = 0; p < queue.size(); p++) {
    NodeDef* cur = queue[p];
    size_t cur_level = level[cur];
    if (match_func(cur)) {
      if (clusters->size() <= cur_level) {
        clusters->resize(cur_level + 1);
      }
      (*clusters)[cur_level].insert(cur);
      cur_level = cur_level + 1;
    }
    for (auto output : outputs[cur]) {
      if (--ref[output] == 0) {
        queue.push_back(output);
      }
      level[output] = std::max(cur_level, level[output]);
    }
  }
  return Status::Ok();
}


Status FusionWorker::MarkDeleteNode(
      const std::set<NodeDef*>& cluster, 
      const NodeDef& fused_node) {
  for (size_t i = 0; i < graph_->node.size(); ++i) {
    if (cluster.find(&(graph_->node[i])) != cluster.end()) {
      graph_->node[i].name = "__delete__";
    }
  }

  new_nodes_.emplace_back(new NodeDef(fused_node));
  nodes_.insert(std::make_pair(fused_node.name, new_nodes_.back().get()));

  return Status::Ok();
}

Status FusionWorker::DeleteNodes() {
  std::vector<NodeDef> nodes;
  for (auto& item: graph_->node) {
    if (item.name != "__delete__") {
      nodes.push_back(item);
    }
  }
  for (auto&& item : new_nodes_) {
    nodes.push_back(*item.get());
  }

  graph_->node = std::move(nodes);
  return Status::Ok();
}

Status FusionWorker::MarkRenameInput(
    const std::string& in_name, const std::string& out_name) {
  rename_map_[in_name] = out_name;
  return Status::Ok();
}

Status FusionWorker::RenameInput() {
  for (auto& item: graph_->node) {
    for (auto& input: item.input) {
      auto iter = rename_map_.find(input);
      if (iter != rename_map_.end()) {
        input = iter->second;
      }
    }
  }
  for (auto& input: output_names_) {
    auto iter = rename_map_.find(input);
    if (iter != rename_map_.end()) {
      input = iter->second;
    }
  }
  output_->output = std::move(output_names_);
  return Status::Ok();
}

}  // namespace xdl




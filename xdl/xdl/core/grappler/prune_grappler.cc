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

#include "xdl/core/framework/grappler.h"

namespace xdl {

namespace {

class PruneExecutor {
 public:
  PruneExecutor(const InputSpec& input, 
                GraphDef* graph, 
                OutputSpec* output)
    : graph_(graph), 
      output_(output) {
    inputs_.insert(input.begin(), input.end());
  }

  Status Run() {
    XDL_CHECK_STATUS(Init());
    if (!inputs_.empty()) {
      XDL_CHECK_STATUS(InitNodeDepOutputs());      
    }

    XDL_CHECK_STATUS(Bfs());
    XDL_CHECK_STATUS(Build());
    return Status::Ok();
  }

 private:
  Status Init() {
    for (auto&& item : graph_->node) {
      node_def_[item.name] = &item;
    }
    XDL_CHECK_STATUS(AppendBfsNode(output_->output));
    return Status::Ok();
  }

  Status GetNodeByName(
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

    auto iter = node_def_.find(node_name);
    XDL_CHECK_COND(iter != node_def_.end(),
                   Status::ArgumentError("not found node:" + name));
    *node = iter->second;
    return Status::Ok();
  }

  Status InitNodeDepOutputs() {
    for (auto& item: graph_->node) {
      for (auto& in: item.input) {
        NodeDef* n = nullptr;
        XDL_CHECK_STATUS(GetNodeByName(in, &n));
        node_dep_outputs_[n].insert(in);
      }
    }

    return Status::Ok();
  }

  Status Bfs() {
    size_t i = 0;
    while (i < bfs_list_.size()) {
      XDL_CHECK_STATUS(AppendBfsNode(bfs_list_[i]->input));
      i++;
    }
    return Status::Ok();
  }

  Status AppendBfsNode(const std::vector<std::string>& names) {
    for (auto&& name : names) {
      std::string real_name;
      if (name.size() > 0 && name[0] == '^') {
        real_name = name.substr(1);
      } else {
        size_t pos = name.find(':');
        XDL_CHECK_COND(pos != std::string::npos,
                       Status::ArgumentError("Node Input Error, "
                                             ": not found " + name));
        real_name = name.substr(0, pos);
      }
      if (avaliable_node_.find(real_name) != avaliable_node_.end()) {
        continue;
      }
      auto iter = node_def_.find(real_name);
      XDL_CHECK_COND(iter != node_def_.end(),
                     Status::ArgumentError("node def has invalid input"));
      avaliable_node_.insert(real_name);
      if (inputs_.find(name) != inputs_.end()) {
        if (!IsFeededAll(iter->second)) {
          return Status::ArgumentError(
              "all outputs of node:" + iter->second->name + " must be feeded");
        }
      } else {
        bfs_list_.push_back(iter->second);
      }
    }

    return Status::Ok();
  }

  bool IsFeededAll(NodeDef* n) {
    auto& output_nodes = node_dep_outputs_[n];
    for (auto& item: output_nodes) {
      if (inputs_.find(item) == inputs_.end()) {
        return false;
      }
    }
    
    return true;
  }

  Status Build() {
    std::vector<NodeDef> nodes;
    for (auto ptr : bfs_list_) {
      nodes.push_back(*ptr);
    }
    graph_->node = std::move(nodes);
    return Status::Ok();
  }

  GraphDef *graph_;
  std::set<std::string> inputs_;
  OutputSpec *output_;
  std::unordered_map<std::string, NodeDef*> node_def_;
  std::unordered_set<std::string> avaliable_node_;
  std::unordered_map<NodeDef*, std::set<std::string> > node_dep_outputs_;
  std::vector<NodeDef*> bfs_list_;
};

}  // namespace


class PruneGrappler : public Grappler {
 public:
  Status Process(
      const InputSpec& input, 
      GraphDef* graph, 
      OutputSpec* output) override {
    PruneExecutor executor(input, graph, output);
    XDL_CHECK_STATUS(executor.Run());
    return Status::Ok();
  }
};

}  // namespace xdl

XDL_REGISTER_GRAPPLER(10000, xdl::PruneGrappler);


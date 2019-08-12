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
  PruneExecutor(GraphDef* graph, OutputSpec* output)
    : graph_(graph), output_(output) {}
  Status Run() {
    XDL_CHECK_STATUS(Init());
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
                     Status::ArgumentError("not found node:" + real_name));
      avaliable_node_.insert(real_name);
      bfs_list_.push_back(iter->second);
    }
    return Status::Ok();
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
  OutputSpec *output_;
  std::unordered_map<std::string, NodeDef*> node_def_;
  std::unordered_set<std::string> avaliable_node_;
  std::vector<NodeDef*> bfs_list_;
};

}  // namespace


class PruneGrappler : public Grappler {
 public:
  Status Process(GraphDef* graph, OutputSpec* output) override {
    PruneExecutor executor(graph, output);
    XDL_CHECK_STATUS(executor.Run());
    return Status::Ok();
  }
};

}  // namespace xdl

XDL_REGISTER_GRAPPLER(10000, xdl::PruneGrappler);


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

#include "udf.h"

namespace ps {
namespace client {

namespace {
// Some constant about hash algorithm
const unsigned long long kP = 98765431;
const unsigned long long kP2 = kP * kP;
const unsigned long long kHash0 = 19940319;
const unsigned long long kHash1 = kP - kHash0;
}

void UdfChain::CalculateHash() {
  hash_ = kHash0 * kP + std::hash<std::string>()("__INPUT__");
  node_ids_[nullptr] = 0;
  nodes_.push_back(nullptr);

  unsigned long long output_hash = 0;
  unsigned long long output_hash_offset = 1;
  for (const auto& data : datas_) {
    output_hash = output_hash * kP2 + CollectNode(data);
    output_hash_offset *= kP2;
  }
  hash_ = hash_ * kP + datas_.size() + kHash1;
  hash_ = hash_ * output_hash_offset + output_hash;
}

// return the hash of {nodex_inputx_node, nodex_inputx_index}
unsigned long long UdfChain::CollectNode(const UdfData& data) {
  UdfData::UdfNode* node = data.node_.get();
  auto iter = node_ids_.find(node);
  if (node_ids_.find(node) != node_ids_.end()) {
    return iter->second * kP + data.id_;
  } else {
    int64_t my_hash = (node->inputs_.size() + kHash0) * kP + std::hash<std::string>()(node->udf_name_);
    int64_t my_hash_offset = kP2;
    for (const auto& child : node->inputs_) {
      my_hash = my_hash * kP2 + CollectNode(child);
      my_hash_offset *= kP2;
    }
    hash_ = hash_ * my_hash_offset + my_hash;
    unsigned long long ret = nodes_.size() * kP + data.id_;
    node_ids_[node] = nodes_.size();
    nodes_.push_back(node);
    return ret;
  }
}

UdfChainRegister UdfChain::BuildChainRegister() const {
  UdfChainRegister result;
  result.hash = hash_;
  for (auto node : nodes_) {
    UdfChainRegister::UdfDef def;
    if (node == nullptr) {
      def.udf_name = "";
    } else {
      def.udf_name = node->udf_name_;
      for (const auto& child : node->inputs_) {
        def.inputs.emplace_back(node_ids_.find(child.node_.get())->second, child.id_);
      }
    }
    result.udfs.push_back(def);
  }
  for (const auto& data : datas_) {
    result.outputs.emplace_back(node_ids_.find(data.node_.get())->second, data.id_);
  }
  return result;
}

}
}

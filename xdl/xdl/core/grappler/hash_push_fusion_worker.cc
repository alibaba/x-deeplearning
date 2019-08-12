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

#include "xdl/core/grappler/hash_push_fusion_worker.h"

namespace xdl {

Status HashPushFusionWorker::Process(
    GraphDef* graph, OutputSpec* output) {
  XDL_CHECK_STATUS(Init(graph, output));
  std::vector<std::set<NodeDef*> > clusters;
  XDL_CHECK_STATUS(
      ClusterNodes(std::bind(&HashPushFusionWorker::NodeMatcher, 
                             this, 
                             std::placeholders::_1),
                   graph, 
                   &clusters));
  std::vector<std::set<NodeDef*> > post_clusters;
  XDL_CHECK_STATUS(PostCluster(clusters, &post_clusters));
  XDL_CHECK_STATUS(DoFusion(post_clusters));
  XDL_CHECK_STATUS(DeleteNodes());
  XDL_CHECK_STATUS(RenameInput());
  return Status::Ok();
}

Status HashPushFusionWorker::PostCluster(
    const std::vector<std::set<NodeDef*> >& clusters,
    std::vector<std::set<NodeDef*> >* sub_clusters) {
  int cluster_id = 0;
  std::vector<std::set<NodeDef*> > tmp;
  for (auto& cluster: clusters) {
    std::map<std::pair<int, std::string>, int> type_2_cluster_id;
    for (auto& node: cluster) {
      DataType itype;
      XDL_CHECK_STATUS(
          GetAttrValue<DataType>(node, "dtype", &itype)); 
      std::pair<int, std::string> type = std::make_pair(itype, node->op);
      auto it = type_2_cluster_id.find(type);
      if (it != type_2_cluster_id.end()) {
        tmp[it->second].insert(node);
      } else {
        tmp.push_back(std::set<NodeDef*>());
        type_2_cluster_id.insert({type, cluster_id++});
        tmp.back().insert(node);
      }
    }
  }

  for (auto&& cluster: tmp) {
    if (cluster.size() > 1) { 
      sub_clusters->emplace_back(cluster);
    }
  }

  // std::cout << "post cluster size:" << sub_clusters->size() << std::endl;
  // for (auto& cluster: (*sub_clusters)) {
  //   std::cout << "cluster:" << std::endl;
  //   for (auto& item: cluster) {
  //     std::cout << item->name << std::endl;
  //   }
  // }
  return Status::Ok();
}

Status HashPushFusionWorker::DoFusion(
    const std::vector<std::set<NodeDef*> >& clusters) {
  for (auto& cluster: clusters) {
    XDL_CHECK_STATUS(FuseOneCluster(cluster));
  }

  return Status::Ok();
}

Status HashPushFusionWorker::FuseOneCluster(
    const std::set<NodeDef*>& cluster) {
  NodeDef* n = *(cluster.begin());
  DataType itype;
  XDL_CHECK_STATUS(
      GetAttrValue<DataType>(n, "dtype", &itype));   
  std::string op_name = n->op;
  std::string var_name_str;
  for (auto& item: cluster) {
    std::string var_name;
    XDL_CHECK_STATUS(
        GetAttrValue<std::string>(item, "var_name", &var_name));    
    var_name_str += var_name + ",";
  }
  var_name_str.pop_back();

  NodeDef fused_node;
  XDL_CHECK_STATUS(
      FuseImpl(var_name_str, op_name,
               itype,
               cluster,
               &fused_node));
  XDL_CHECK_STATUS(
      MarkDeleteNode(cluster, fused_node));
  return Status::Ok();
}

namespace {
struct FusionStrategy {
  std::string op;
  std::vector<DataType> input_types;
};

const std::unordered_map<std::string, FusionStrategy> kFusionStrategies = {
  {"PsSparseApplyAdagradOp", {.op="PsSparseApplyAdagradMergedOp",
   .input_types={DataType::kDouble, DataType::kDouble, DataType::kFloat}}},
  {"PsSparseApplyAdamOp", {.op="PsSparseApplyAdamMergedOp",
   .input_types={DataType::kDouble, DataType::kDouble, DataType::kDouble,
                 DataType::kDouble, DataType::kBool, DataType::kFloat}}},
  {"PsSparseApplyFtrlOp", {.op="PsSparseApplyFtrlMergedOp",
   .input_types={DataType::kDouble, DataType::kDouble, DataType::kDouble,
                 DataType::kDouble, DataType::kDouble, DataType::kFloat}}},
  {"PsSparseApplyMomentumOp", {.op="PsSparseApplyMomentumMergedOp",
   .input_types={DataType::kDouble, DataType::kDouble,
                 DataType::kBool, DataType::kFloat}}},
  {"PsSparseApplyRmspropOp", {.op="PsSparseApplyRmspropMergedOp",
   .input_types={DataType::kDouble, DataType::kDouble, DataType::kDouble,
                 DataType::kDouble, DataType::kFloat}}}
};
}

Status HashPushFusionWorker::FuseImpl(
    const std::string& var_name_str,
    const std::string& op_name,
    DataType itype,
    const std::set<NodeDef*>& cluster,
    NodeDef* node) {
  auto iter = kFusionStrategies.find(op_name);
  if (iter == kFusionStrategies.end()) {
    return Status::ArgumentError("Fused op_name " + op_name + " unsupported");
  }
  const FusionStrategy& strategy = iter->second;
  if (id_ == 0) {
    node->name = strategy.op;
  } else {
    node->name = strategy.op + "_" + std::to_string(id_);
  }
  id_++;

  node->op = strategy.op;

  std::vector<DataType> dtypes(strategy.input_types.begin(), strategy.input_types.end());
  dtypes.push_back(itype);

  std::vector<std::vector<DataType>> input_types(strategy.input_types.size() + 1);
  std::vector<std::vector<std::string>> inputs(strategy.input_types.size() + 1);
  std::vector<std::string> dependencies;
  int output_idx = 0;

  for (auto& item: cluster) {
    std::vector<std::string> non_depend_input;
    for (auto& input : item->input) {
      if (input.size() > 0 && input[0] == '^') {
        dependencies.push_back(input);
      } else {
        non_depend_input.push_back(input);
      }
    }
    XDL_CHECK_COND(non_depend_input.size() == strategy.input_types.size() + 1,
                   Status::ArgumentError(item->name + " Input Error, not a " + op_name));

    for (size_t i = 0; i < strategy.input_types.size() + 1; i++) {
      input_types[i].push_back(dtypes[i]);
      inputs[i].push_back(non_depend_input[i]);
    }

    MarkRenameInput("^" + item->name, "^" + node->name);
  }

  for (size_t i = 0; i < strategy.input_types.size() + 1; i++) {
    SetAttrValue<std::vector<DataType>>(
      node, "input_type_" + std::to_string(i), input_types[i]);
    node->input.insert(node->input.end(), inputs[i].begin(), inputs[i].end());
  }
  node->input.insert(node->input.end(), dependencies.begin(), dependencies.end());
  SetAttrValue<std::string>(node, "var_type", "hash");
  SetAttrValue<std::string>(node, "var_name", "hash_variable");
  SetAttrValue<std::string>(node, "var_names", var_name_str);
  node->device.device_name = "CPU";
  return Status::Ok();
}

} //namespace xdl

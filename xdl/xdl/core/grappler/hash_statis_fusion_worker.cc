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

#include "xdl/core/grappler/hash_statis_fusion_worker.h"

namespace xdl {

Status HashStatisFusionWorker::Process(
    GraphDef* graph, OutputSpec* output) {
  XDL_CHECK_STATUS(Init(graph, output));
  std::vector<std::set<NodeDef*> > clusters;
  XDL_CHECK_STATUS(
      ClusterNodes(std::bind(&HashStatisFusionWorker::NodeMatcher,
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

Status HashStatisFusionWorker::PostCluster(
    const std::vector<std::set<NodeDef*> >& clusters,
    std::vector<std::set<NodeDef*> >* sub_clusters) {
  int cluster_id = 0;
  std::vector<std::set<NodeDef*> > tmp;
  for (auto& cluster: clusters) {
    std::map<std::pair<int, int>, int> type_2_cluster_id;
    for (auto& node: cluster) {
      DataType itype;
      XDL_CHECK_STATUS(
          GetAttrValue<DataType>(node, "dtype", &itype));    
      DataType otype;
      XDL_CHECK_STATUS(
          GetAttrValue<DataType>(node, "otype", &otype));    
      std::pair<int, int> type = std::make_pair(itype, otype);
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
    if (cluster.size() >= 1) {
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

Status HashStatisFusionWorker::DoFusion(
    const std::vector<std::set<NodeDef*> >& clusters) {
  for (auto& cluster: clusters) {
    XDL_CHECK_STATUS(FuseOneCluster(cluster));
  }

  return Status::Ok();
}

Status HashStatisFusionWorker::FuseOneCluster(
    const std::set<NodeDef*>& cluster) {
  NodeDef* n = *(cluster.begin());
  DataType itype;
  XDL_CHECK_STATUS(
      GetAttrValue<DataType>(n, "dtype", &itype));    
  DataType otype;
  XDL_CHECK_STATUS(
      GetAttrValue<DataType>(n, "otype", &otype));    
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
      FuseImpl(var_name_str,
               itype, otype, 
               cluster,
               &fused_node));
  XDL_CHECK_STATUS(
      MarkDeleteNode(cluster, fused_node));
  return Status::Ok();
}

Status HashStatisFusionWorker::FuseImpl(
    const std::string& var_name_str,
    DataType itype,
    DataType otype,
    const std::set<NodeDef*>& cluster,
    NodeDef* node) {
  node->name = "PsMergedSparseStatisOp_" + statis_type_ + "_" + std::to_string(id_++);
  node->op = "PsMergedSparseStatisOp";
  int output_idx = 0;
  std::vector<DataType> input_type, input_type_1, input_type_2, input_type_3, input_type_4, input_type_5, input_type_6;
  std::vector<std::vector<std::string>> inputs;
  const size_t input_lists_size = 7;
  inputs.resize(input_lists_size);
  std::vector<std::string> dependencies;

  for (auto& item: cluster) {
    std::vector<std::string> non_depend_input;
    for (auto& input : item->input) {
      if (input.size() > 0 && input[0] == '^') {
        dependencies.push_back(input);
      } else {
        non_depend_input.push_back(input);
      }
    }
    XDL_CHECK_COND(non_depend_input.size() == 10,
                   Status::ArgumentError(item->name + " Input Error, not a PsSparseStatisOp"));

    for (size_t i = 0; i < input_lists_size; ++i) {
      inputs[i].push_back(non_depend_input[i]);
      MarkRenameInput(item->name + ":" + std::to_string(i), node->name + ":" + std::to_string(output_idx));
    }
    MarkRenameInput("^" + item->name, "^" + node->name);

    output_idx++;
    input_type.push_back(itype);
    input_type_1.push_back(DataType::kInt32);
    input_type_2.push_back(DataType::kInt32);
    input_type_3.push_back(DataType::kInt32);
    input_type_4.push_back(DataType::kInt32);
    input_type_5.push_back(DataType::kFloat);
    input_type_6.push_back(DataType::kFloat);
  }
  for (size_t i = 0; i < input_lists_size; ++i) {
    node->input.insert(node->input.end(), inputs[i].begin(), inputs[i].end());
  }
  for (auto& item: cluster) {
    std::vector<std::string> non_depend_input;
    for (auto& input : item->input) {
      if (input.size() > 0 && input[0] == '^') {
        dependencies.push_back(input);
      } else {
        non_depend_input.push_back(input);
      }
    }
    for (size_t i = input_lists_size; i < non_depend_input.size(); ++i) {
      node->input.push_back(non_depend_input[i]);
    }
    break;
  }
  node->input.insert(node->input.end(), dependencies.begin(), dependencies.end());

  SetAttrValue<std::vector<DataType>>(node, "input_type", input_type);
  SetAttrValue<std::vector<DataType>>(node, "input_type_1", input_type_1);
  SetAttrValue<std::vector<DataType>>(node, "input_type_2", input_type_2);
  SetAttrValue<std::vector<DataType>>(node, "input_type_3", input_type_3);
  SetAttrValue<std::vector<DataType>>(node, "input_type_4", input_type_4);  
  SetAttrValue<std::vector<DataType>>(node, "input_type_5", input_type_5);
  SetAttrValue<std::vector<DataType>>(node, "input_type_6", input_type_6);
  SetAttrValue<std::string>(node, "statis_type", statis_type_);
  SetAttrValue<std::string>(node, "var_type", "hash");
  SetAttrValue<std::string>(node, "var_name", "hash_variable");
  SetAttrValue<std::string>(node, "var_names", var_name_str);
  SetAttrValue<int>(node, "output_size", input_type.size());
  node->device.device_name = "CPU";
  return Status::Ok();
}

} //namespace xdl

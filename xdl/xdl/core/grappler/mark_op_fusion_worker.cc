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

#include "xdl/core/grappler/mark_op_fusion_worker.h"

namespace xdl {

Status MarkOpFusionWorker::Process(
    GraphDef* graph, OutputSpec* output) {
  XDL_CHECK_STATUS(Init(graph, output));
  std::vector<std::set<NodeDef*> > clusters;
  XDL_CHECK_STATUS(
      ClusterNodes(std::bind(&MarkOpFusionWorker::NodeMatcher, 
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

Status MarkOpFusionWorker::PostCluster(
    const std::vector<std::set<NodeDef*> >& clusters,
    std::vector<std::set<NodeDef*> >* sub_clusters) {
  int cluster_id = 0;
  std::vector<std::set<NodeDef*> > tmp;
  for (auto& cluster: clusters) {
    std::map<int, int> type_2_cluster_id;
    for (auto& node: cluster) {
      DataType itype;
      XDL_CHECK_STATUS(
          GetAttrValue<DataType>(node, "dtype", &itype));      
      auto it = type_2_cluster_id.find(itype);
      if (it != type_2_cluster_id.end()) {
        tmp[it->second].insert(node);
      } else {
        tmp.push_back(std::set<NodeDef*>());
        type_2_cluster_id.insert({itype, cluster_id++});
        tmp.back().insert(node);
      }
    }
  }

  for (auto&& cluster: tmp) {
    if (cluster.size() > 1) { 
      sub_clusters->emplace_back(cluster);
    }
  }

  return Status::Ok();
}

Status MarkOpFusionWorker::DoFusion(
    const std::vector<std::set<NodeDef*> >& clusters) {
  for (auto& cluster: clusters) {
    XDL_CHECK_STATUS(FuseOneCluster(cluster));
  }

  return Status::Ok();
}

Status MarkOpFusionWorker::FuseOneCluster(
    const std::set<NodeDef*>& cluster) {
  NodeDef* n = *(cluster.begin());
  DataType itype;
  XDL_CHECK_STATUS(
      GetAttrValue<DataType>(n, "dtype", &itype));    
  std::string var_name_str;
  for (auto& item: cluster) {
    std::string var_name;
    XDL_CHECK_STATUS(
        GetAttrValue<std::string>(item, "var_name", &var_name));    
    var_name_str += var_name + ",";
  }
  var_name_str.pop_back();

  std::string pattern_str;
  for (auto& item: cluster) {
    std::string pattern;
    XDL_CHECK_STATUS(
        GetAttrValue<std::string>(item, "pattern", &pattern));    
    pattern_str += pattern + ",";
  }
  pattern_str.pop_back();

  NodeDef fused_node;
  XDL_CHECK_STATUS(
      FuseImpl(var_name_str,
               pattern_str,
               itype,
               cluster,
               &fused_node));
  XDL_CHECK_STATUS(
      MarkDeleteNode(cluster, fused_node));
  return Status::Ok();
}

Status MarkOpFusionWorker::FuseImpl(
    const std::string& var_name_str,
    const std::string& pattern_str,
    DataType itype,
    const std::set<NodeDef*>& cluster,
    NodeDef* node) {
  if (id_ == 0) {
    node->name = "PsMergedMarkOp";
    id_++;
  } else {
    node->name = "PsMergedMarkOp_" + std::to_string(id_++);
  }

  node->op = "PsMergedMarkOp";
  std::vector<DataType> input_type_0;
  std::vector<DataType> input_type_1;
  std::vector<std::string> input_0;
  std::vector<std::string> input_1;
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
    XDL_CHECK_COND(non_depend_input.size() == 2,
                   Status::ArgumentError(item->name + " Input Error, not a PsMarkOp"));

    input_0.push_back(non_depend_input[0]);
    input_1.push_back(non_depend_input[1]);
    MarkRenameInput("^" + item->name, "^" + node->name);

    input_type_0.push_back(itype);
    input_type_1.push_back(DataType::kInt64);
  }
  node->input.insert(node->input.end(), input_0.begin(), input_0.end());
  node->input.insert(node->input.end(), input_1.begin(), input_1.end());
  node->input.insert(node->input.end(), dependencies.begin(), dependencies.end());

  SetAttrValue<std::vector<DataType> >(
      node, "input_type_0", input_type_0);
  SetAttrValue<std::vector<DataType> >(
      node, "input_type_1", input_type_1);
  SetAttrValue<std::string>(node, "var_names", var_name_str);
  SetAttrValue<std::string>(node, "patterns", pattern_str);
  node->device.device_name = "CPU";
  return Status::Ok();
}

} //namespace xdl


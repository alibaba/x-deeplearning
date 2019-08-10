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

#ifndef XDL_CORE_GRAPPLER_PS_FUSION_WORKER
#define XDL_CORE_GRAPPLER_PS_FUSION_WORKER

#include <atomic>
#include <set>
#include <map>
#include <vector>
#include <functional>
#include <unordered_map>

#include "xdl/core/framework/graph_def.h"
#include "xdl/core/lib/status.h"

namespace xdl {

class FusionWorker {
 public:
  using NodeDependency = std::unordered_map<NodeDef*, std::set<NodeDef*> >;
  using NodeInput = std::pair<NodeDef*, int>;  
  using PredecessorMap = std::unordered_map<NodeDef*, std::set<NodeInput> >;

  FusionWorker() : id_(0) {}
  virtual ~FusionWorker() {}
  virtual Status Process(GraphDef* graph, OutputSpec* output) = 0;
  
protected:
  Status Init(GraphDef* graph, OutputSpec* output);
  Status ClusterNodes(
      const std::function<bool(NodeDef*)>& match_func,
      GraphDef* graph, 
      std::vector<std::set<NodeDef*> >* clusters);
  Status GetNodeByName(
      const std::string& name, 
      NodeDef** node);
  Status GetNodeAttr(
      NodeDef* node, 
      const std::string& name,
      AttrValue* value);
  template <typename T>
  Status GetAttrValue(
      NodeDef* node, 
      const std::string& name,
      T* value);
  template <typename T>
  void SetAttrValue(
      NodeDef* node, 
      const std::string& name,
      const T& value);
  Status MarkDeleteNode(
      const std::set<NodeDef*>& cluster, 
      const NodeDef& fused_node);
  Status DeleteNodes();
  Status MarkRenameInput(
      const std::string& in_name, const std::string& out_name);
  Status RenameInput();
  
 protected:
  std::unordered_map<std::string, NodeDef*> nodes_;
  std::unordered_map<std::string, std::string> rename_map_;
  std::vector<NodeDef*> output_nodes_;
  std::vector<std::string> output_names_;
  std::atomic<int> id_;
  std::vector<std::unique_ptr<NodeDef>> new_nodes_;
  GraphDef* graph_;
  OutputSpec* output_;
};

template <>
inline Status FusionWorker::GetAttrValue(
    NodeDef* node,
    const std::string& name,
    std::string* attr) {
  AttrValue value;
  XDL_CHECK_STATUS(GetNodeAttr(node, name, &value));
  XDL_CHECK_COND(value.attr_type == AttrValue::kString, 
                 Status::ArgumentError("attr:" + name + " is not string"));
  *attr = value.s;
  return Status::Ok();
}

template <>
inline Status FusionWorker::GetAttrValue(
    NodeDef* node,
    const std::string& name,
    int* attr) {
  AttrValue value;
  XDL_CHECK_STATUS(GetNodeAttr(node, name, &value));
  XDL_CHECK_COND(value.attr_type == AttrValue::kInt, 
                 Status::ArgumentError("attr:" + name + " is not int"));
  *attr = value.i;
  return Status::Ok();
}

template <>
inline Status FusionWorker::GetAttrValue(
    NodeDef* node,
    const std::string& name,
    float* attr) {
  AttrValue value;
  XDL_CHECK_STATUS(GetNodeAttr(node, name, &value));
  XDL_CHECK_COND(value.attr_type == AttrValue::kFloat, 
                 Status::ArgumentError("attr:" + name + " is not float"));
  *attr = value.f;
  return Status::Ok();
}

template <>
inline Status FusionWorker::GetAttrValue(
    NodeDef* node,
    const std::string& name,
    bool* attr) {
  AttrValue value;
  XDL_CHECK_STATUS(GetNodeAttr(node, name, &value));
  XDL_CHECK_COND(value.attr_type == AttrValue::kBool, 
                 Status::ArgumentError("attr:" + name + " is not bool"));
  *attr = value.b;
  return Status::Ok();
}

template <>
inline Status FusionWorker::GetAttrValue(
    NodeDef* node,
    const std::string& name,
    DataType* attr) {
  AttrValue value;
  XDL_CHECK_STATUS(GetNodeAttr(node, name, &value));
  XDL_CHECK_COND(value.attr_type == AttrValue::kDataType, 
                 Status::ArgumentError("attr:" + name + " is not DataType"));
  *attr = value.type;
  return Status::Ok();
}

template <>
inline Status FusionWorker::GetAttrValue(
    NodeDef* node,
    const std::string& name,
    TensorShape* attr) {
  AttrValue value;
  XDL_CHECK_STATUS(GetNodeAttr(node, name, &value));
  XDL_CHECK_COND(value.attr_type == AttrValue::kTensorShape, 
                 Status::ArgumentError("attr:" + name + " is not TensorShape"));
  *attr = value.shape;
  return Status::Ok();
}

template <>
inline void FusionWorker::SetAttrValue(
    NodeDef* node,
    const std::string& name,
    const std::string& attr) {
  AttrValue value;
  value.attr_type = AttrValue::kString;
  value.s = attr;
  node->attr[name] = value;
}

template <>
inline void FusionWorker::SetAttrValue(
    NodeDef* node,
    const std::string& name,
    const int& attr) {
  AttrValue value;
  value.attr_type = AttrValue::kInt;
  value.i = attr;
  node->attr[name] = value;
}

template <>
inline void FusionWorker::SetAttrValue(
    NodeDef* node,
    const std::string& name,
    const float& attr) {
  AttrValue value;
  value.attr_type = AttrValue::kFloat;
  value.f = attr;
  node->attr[name] = value;
}

template <>
inline void FusionWorker::SetAttrValue(
    NodeDef* node,
    const std::string& name,
    const bool& attr) {
  AttrValue value;
  value.attr_type = AttrValue::kBool;
  value.b = attr;
  node->attr[name] = value;
}

template <>
inline void FusionWorker::SetAttrValue(
    NodeDef* node,
    const std::string& name,
    const DataType& attr) {
  AttrValue value;
  value.attr_type = AttrValue::kDataType;
  value.type = attr;
  node->attr[name] = value;
}

template <>
inline void FusionWorker::SetAttrValue(
    NodeDef* node,
    const std::string& name,
    const TensorShape& attr) {
  AttrValue value;
  value.attr_type = AttrValue::kTensorShape;
  value.shape = attr;
  node->attr[name] = value;
}

template <>
inline void FusionWorker::SetAttrValue(
    NodeDef* node,
    const std::string& name,
    const std::vector<DataType>& attr) {
  AttrValue value;
  value.attr_type = AttrValue::kDataTypeList;
  value.type_list = attr;
  node->attr[name] = value;
}

inline Status FusionWorker::GetNodeAttr(
    NodeDef* node, 
    const std::string& name,
    AttrValue* value) {
  const auto& it = node->attr.find(name);
  XDL_CHECK_COND(it != node->attr.end(), 
                 Status::ArgumentError("attr:" + name + " not found"));
  *value = it->second;
  return Status::Ok();
}

}  // namespace XDL_CORE_GRAPPLER_PS_FUSION_WORKER

#endif


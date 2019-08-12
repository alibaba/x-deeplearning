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

#ifndef XDL_CORE_FRAMEWORK_GRAPH_DEF_H_
#define XDL_CORE_FRAMEWORK_GRAPH_DEF_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "xdl/core/framework/types.h"
#include "xdl/core/framework/tensor_shape.h"

#include "xdl/core/proto/graph_def.pb.h"

namespace xdl {

struct DeviceDef {
  std::string device_name;
  std::unordered_map<std::string, std::string> attr;

  proto::DeviceDef ToProto() const;
  void FromProto(const proto::DeviceDef& pb);
};

struct AttrValue {
  enum Type {
    kNone,
    kString,
    kInt,
    kFloat,
    kBool,
    kDataType,
    kTensorShape,
    kDataTypeList
  };

  Type attr_type;
  std::string s;
  int64_t i;
  float f;
  bool b;
  DataType type;
  TensorShape shape;
  std::vector<DataType> type_list;

  proto::AttrValue ToProto() const;
  void FromProto(const proto::AttrValue& pb);
};

struct NodeDef {
  std::string name;
  std::string op;
  std::vector<std::string> input;
  DeviceDef device;
  std::vector<std::string> input_dev_descs;  // NOT IN PROTO
  std::unordered_map<std::string, AttrValue> attr;
  std::vector<DataType> output_type;

  proto::NodeDef ToProto() const;
  void FromProto(const proto::NodeDef& pb);
};

struct GraphDef {
  std::vector<NodeDef> node;
  int64_t hash;
  proto::GraphDef ToProto() const;
  void FromProto(const proto::GraphDef& pb);
  std::string ToProtoString() const;
  void FromProtoTxtString(const std::string& pb_string);
};

struct OutputSpec {
  std::vector<std::string> output;
  DeviceDef output_device;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_GRAPH_DEF_H_


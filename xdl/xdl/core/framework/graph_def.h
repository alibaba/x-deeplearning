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
  std::unordered_map<std::string, AttrValue> attr;

  proto::NodeDef ToProto() const;
  void FromProto(const proto::NodeDef& pb);
};

enum InputType {
  kInputSparse = 0,
  kInputDense,
  kInputOther
};

struct InputDef {
  std::string op_name;
  std::string input_name;
  InputType input_type;
  int size;
  int table;
  std::vector<uint32_t> mask;
  void FromProto(const proto::InputDef& def) {
    op_name = def.op_name();
    input_name = def.input_name();
    input_type = static_cast<InputType>(def.input_type());
    size = def.size();
    table = def.table();
    for (size_t i = 0; i < def.mask_size(); ++i) {
      mask.push_back(def.mask(i));
    }
  }

  proto::InputDef ToProto() const {
    proto::InputDef proto;
    proto.set_op_name(op_name);
    proto.set_input_name(input_name);
    proto.set_input_type(static_cast<proto::InputType>(input_type));
    proto.set_size(size);
    proto.set_table(table);
    for (auto& item: mask) {
      proto.add_mask(item);
    }

    return proto;
  }
};

struct OutputDef {
  std::string op_name;
  void FromProto(const proto::OutputDef& def) {
    op_name = def.op_name();
  }

  proto::OutputDef ToProto() const {
    proto::OutputDef proto;
    proto.set_op_name(op_name);
    return proto;
  }
};

struct TagDef {
  std::vector<InputDef> inputs;
  std::vector<OutputDef> outputs;
  void FromProto(const proto::TagDef& def) {
    for (size_t i = 0; i < def.input_size(); ++i) {
      inputs.emplace_back();
      inputs.back().FromProto(def.input(i));
    }

    for (size_t i = 0; i < def.output_size(); ++i) {
      outputs.emplace_back();
      outputs.back().FromProto(def.output(i));
    }
  }

  proto::TagDef ToProto() const {
    proto::TagDef proto;    
    for (auto& item: inputs) {
      *(proto.add_input()) = item.ToProto();
    }

    for (auto& item: outputs) {
      *(proto.add_output()) = item.ToProto();
    }

    return proto;
  }
};

struct GraphDef {
  std::vector<NodeDef> node;
  int64_t hash;
  TagDef tag;
  proto::GraphDef ToProto() const;
  void FromProto(const proto::GraphDef& pb);
  bool FromProtoString(const std::string& pb);
  bool FromTextString(const std::string& txt);
  std::string ToProtoString() const;
};

using InputSpec = std::vector<std::string>;

struct OutputSpec {
  std::vector<std::string> output;
  DeviceDef output_device;
};

struct RunOption {
  RunOption() : perf(false) {}
  RunOption(bool perf) : perf(perf) {}
  bool perf;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_GRAPH_DEF_H_


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

#include "xdl/core/framework/graph_def.h"

#include <google/protobuf/text_format.h>

namespace xdl {

namespace {

proto::DataType DataTypeToProto(const DataType& type) {
  return static_cast<proto::DataType>(static_cast<int32_t>(type));
}

DataType DataTypeFromProto(proto::DataType pb) {
  return static_cast<DataType>(static_cast<int32_t>(pb));
}

proto::TensorShape TensorShapeToProto(const TensorShape& shape) {
  proto::TensorShape result;
  for (auto item : shape.Dims()) {
    result.add_dim(item);
  }
  return result;
}

TensorShape TensorShapeFromProto(proto::TensorShape pb) {
  std::vector<size_t> dims(pb.dim().begin(), pb.dim().end());
  return TensorShape(std::move(dims));
}

}  // namespace

proto::DeviceDef DeviceDef::ToProto() const {
  proto::DeviceDef result;
  result.set_device_name(this->device_name);
  for (auto& item : this->attr) {
    (*result.mutable_attr())[item.first] = item.second;
  }
  return result;
}

void DeviceDef::FromProto(const proto::DeviceDef& pb) {
  this->device_name = pb.device_name();
  this->attr.clear();
  for (auto& item : pb.attr()) {
    this->attr[item.first] = item.second;
  }
}

proto::AttrValue AttrValue::ToProto() const {
  proto::AttrValue result;
  switch (this->attr_type) {
    case kNone:
      break;
    case kString:
      result.set_s(this->s);
      break;
    case kInt:
      result.set_i(this->i);
      break;
    case kFloat:
      result.set_f(this->f);
      break;
    case kBool:
      result.set_b(this->b);
      break;
    case kDataType:
      result.set_type(DataTypeToProto(this->type));
      break;
    case kTensorShape:
      *result.mutable_shape() = TensorShapeToProto(this->shape);
      break;
    case kDataTypeList: {
      for (auto type : this->type_list) {
        result.mutable_type_list()->add_type(DataTypeToProto(type));
      }
      break;
    }
  }
  return result;
}

void AttrValue::FromProto(const proto::AttrValue& pb) {
  switch (pb.value_case()) {
    case proto::AttrValue::VALUE_NOT_SET:
      this->attr_type = kNone;
      break;
    case proto::AttrValue::kS:
      this->attr_type = kString;
      this->s = pb.s();
      break;
    case proto::AttrValue::kI:
      this->attr_type = kInt;
      this->i = pb.i();
      break;
    case proto::AttrValue::kF:
      this->attr_type = kFloat;
      this->f = pb.f();
      break;
    case proto::AttrValue::kB:
      this->attr_type = kBool;
      this->b = pb.b();
      break;
    case proto::AttrValue::kType:
      this->attr_type = kDataType;
      this->type = DataTypeFromProto(pb.type());
      break;
    case proto::AttrValue::kShape:
      this->attr_type = kTensorShape;
      this->shape = TensorShapeFromProto(pb.shape());
      break;
    case proto::AttrValue::kTypeList: {
      this->attr_type = kDataTypeList;
      for (int i = 0; i < pb.type_list().type_size(); i++) {
        this->type_list.push_back(DataTypeFromProto(pb.type_list().type(i)));
      }
      break;
    }
  }
}

proto::NodeDef NodeDef::ToProto() const {
  proto::NodeDef result;
  result.set_name(this->name);
  result.set_op(this->op);
  for (auto&& item : this->input) {
    result.add_input(item);
  }
  for (auto&& item : this->output_type) {
    result.add_output_type(DataTypeToProto(item));
  }
  (*result.mutable_device()) = this->device.ToProto();
  for (auto&& item : this->attr) {
    (*result.mutable_attr())[item.first] = item.second.ToProto();
  }
  return result;
}

void NodeDef::FromProto(const proto::NodeDef& pb) {
  this->name = pb.name();
  this->op = pb.op();
  this->input.clear();
  for (auto&& item : pb.input()) {
    this->input.push_back(item);
  }
  this->output_type.clear();
  for (auto&& item : pb.output_type()) {
    this->output_type.push_back((DataType)(item));
  }
  this->device.FromProto(pb.device());
  this->attr.clear();
  for (auto&& item : pb.attr()) {
    this->attr[item.first].FromProto(item.second);
  }
}

proto::GraphDef GraphDef::ToProto() const {
  proto::GraphDef result;
  for (auto&& item : this->node) {
    (*result.add_node()) = item.ToProto();
  }
  result.set_hash(hash);
  return result;
}

std::string GraphDef::ToProtoString() const {
  std::string buf;
  google::protobuf::TextFormat::PrintToString(ToProto(), &buf);
  return buf;
}

void GraphDef::FromProto(const proto::GraphDef& pb) {
  this->node.clear();
  for (auto&& item : pb.node()) {
    this->node.emplace_back();
    this->node.back().FromProto(item);
  }
  this->hash = pb.hash();
}

void GraphDef::FromProtoTxtString(const std::string& pb_string) {
  proto::GraphDef pb;
  google::protobuf::TextFormat::ParseFromString(pb_string, &pb);
  this->node.clear();
  for (auto&& item : pb.node()) {
    this->node.emplace_back();
    this->node.back().FromProto(item);
  }
  this->hash = pb.hash();
}

}  // namespace xdl


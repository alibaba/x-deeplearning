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

#include "xdl/core/framework/op_define.h"

#include <vector>
#include <iostream>

namespace xdl {
void OpDefine::DefineOp(const std::string& name, OpDefineItem* item) {
  item->status = ValidateDef(item);
  if (!item->status.IsOk()) {
    std::cerr << "Error On Define Op " << name << std::endl
              << item->status.ToString() << std::endl;
    abort();
  }
  if (definitions_.find(name) != definitions_.end()) {
    std::cerr << "Redefined Op " << name << std::endl;
    //abort();
  }
  definitions_[name].reset(item);
}

std::unordered_map<std::string, OpDefineItem*> OpDefine::GetDefinitions() {
  std::unordered_map<std::string, OpDefineItem*> ret;
  for (auto&& item : definitions_) {
    ret[item.first] = item.second.get();
  }
  return ret;
}

Status OpDefine::Validate(
    const std::string& op,
    const std::unordered_map<std::string, AttrValue>& attr) {
  OpDefineItem* def;
  XDL_CHECK_STATUS(GetDef(op, &def));
  XDL_CHECK_STATUS(def->status);
  for (auto&& item : def->attrs) {
    if (item.default_value.attr_type == AttrValue::kNone) {
      auto iter = attr.find(item.name);
      XDL_CHECK_COND(iter != attr.end(),
                     Status::ArgumentError("Attr not found " + item.name));
      XDL_CHECK_COND(iter->second.attr_type == item.type,
                     Status::ArgumentError("Attr type Error " + item.name));
    }
  }
  return Status::Ok();
}

Status OpDefine::PrepareOpKernelContext(
    const std::string& op,
    std::unordered_map<std::string, AttrValue>* attr,
    OpKernelContextArg* arg) {
  OpDefineItem* def;
  XDL_CHECK_STATUS(GetDef(op, &def));
  XDL_CHECK_STATUS(PrepareAttr(def, attr));
  for (auto&& input : def->inputs) {
    switch (input.type.repeated) {
    case OpDefineItem::RepeatType::kNoRepeat: {
      DataType type;
      XDL_CHECK_STATUS(ParseRefDataType(input.type.type, *attr, &type));
      arg->input_id[input.name] = arg->input_name.size();
      arg->input_name.push_back(input.name);
      arg->input_type.push_back(type);
      break;
    }
    case OpDefineItem::RepeatType::kTypeAndSize: {
      DataType type;
      int size;
      XDL_CHECK_STATUS(ParseRefDataType(input.type.type, *attr, &type));
      XDL_CHECK_STATUS(ParseRefInt64(input.type.size, *attr, &size));
      auto& input_list_id = arg->input_list_id[input.name];
      for (int i = 0; i < size; i++) {
        input_list_id.push_back(arg->input_name.size());
        arg->input_name.push_back(input.name);
        arg->input_type.push_back(type);
      }
      break;
    }
    case OpDefineItem::RepeatType::kTypeList: {
      std::vector<DataType> type_list;
      XDL_CHECK_STATUS(
          ParseDataTypeList(input.type.type_list, *attr, &type_list));
      auto& input_list_id = arg->input_list_id[input.name];
      for (auto type : type_list) {
        input_list_id.push_back(arg->input_name.size());
        arg->input_name.push_back(input.name);
        arg->input_type.push_back(type);
      }
      break;
    }
    }
  }
  for (auto&& output : def->outputs) {
    switch (output.type.repeated) {
    case OpDefineItem::RepeatType::kNoRepeat: {
      DataType type;
      XDL_CHECK_STATUS(ParseRefDataType(output.type.type, *attr, &type));
      arg->output_id[output.name] = arg->output_name.size();
      arg->output_name.push_back(output.name);
      arg->output_type.push_back(type);
      break;
    }
    case OpDefineItem::RepeatType::kTypeAndSize: {
      DataType type;
      int size;
      XDL_CHECK_STATUS(ParseRefDataType(output.type.type, *attr, &type));
      XDL_CHECK_STATUS(ParseRefInt64(output.type.size, *attr, &size));
      auto& output_list_id = arg->output_list_id[output.name];
      for (int i = 0; i < size; i++) {
        output_list_id.push_back(arg->output_name.size());
        arg->output_name.push_back(output.name);
        arg->output_type.push_back(type);
      }
      break;
    }
    case OpDefineItem::RepeatType::kTypeList: {
      std::vector<DataType> type_list;
      XDL_CHECK_STATUS(
          ParseDataTypeList(output.type.type_list, *attr, &type_list));
      auto& output_list_id = arg->output_list_id[output.name];
      for (auto type : type_list) {
        output_list_id.push_back(arg->output_name.size());
        arg->output_name.push_back(output.name);
        arg->output_type.push_back(type);
      }
      break;
    }
    }
  }
  return Status::Ok();
}

Status OpDefine::GetTag(const std::string& op, std::vector<std::string>* tag) {
  OpDefineItem* def;
  XDL_CHECK_STATUS(GetDef(op, &def));
  *tag = def->tags;
  return Status::Ok();
}

Status OpDefine::ValidateDef(OpDefineItem* def) {
  std::unordered_set<std::string> dict;
  for (auto&& item : def->inputs) {
    XDL_CHECK_COND(dict.insert(item.name).second,
                   Status::ArgumentError("Redefined op input " + item.name));
  }
  for (auto&& item : def->outputs) {
    XDL_CHECK_COND(dict.insert(item.name).second,
                   Status::ArgumentError("Redefined op output " + item.name));
  }
  for (auto&& item : def->attrs) {
    XDL_CHECK_COND(dict.insert(item.name).second,
                   Status::ArgumentError("Redefined op attr " + item.name));
  }
  std::unordered_map<std::string, OpDefineItem::Attr> attrs;
  for (auto&& item : def->attrs) {
    attrs[item.name] = item;
    if (item.default_value.attr_type != AttrValue::kNone) {
      XDL_CHECK_COND(item.default_value.attr_type == item.type,
                     Status::ArgumentError("Attr Default Value Type Error "
                                           + item.name));
    }
  }
  auto has_attr = [&](const std::string& name, AttrValue::Type t) -> Status {
    auto iter = attrs.find(name);
    XDL_CHECK_COND(iter != attrs.end(),
                   Status::ArgumentError("Attr Not Found " + name));
    XDL_CHECK_COND(iter->second.type == t,
                   Status::ArgumentError("Attr Type Error " + name));
    return Status::Ok();
  };
  for (auto&& item : def->inputs) {
    if (item.type.type.attr != "") {
      XDL_CHECK_STATUS(has_attr(item.type.type.attr, AttrValue::kDataType));
    }
    if (item.type.size.attr != "") {
      XDL_CHECK_STATUS(has_attr(item.type.size.attr, AttrValue::kInt));
    }
  }
  for (auto&& item : def->outputs) {
    if (item.type.type.attr != "") {
      XDL_CHECK_STATUS(has_attr(item.type.type.attr, AttrValue::kDataType));
    }
    if (item.type.size.attr != "") {
      XDL_CHECK_STATUS(has_attr(item.type.size.attr, AttrValue::kInt));
    }
  }
  return Status::Ok();
}

Status OpDefine::PrepareAttr(
    OpDefineItem* def,
    std::unordered_map<std::string, AttrValue>* attr) {
  XDL_CHECK_STATUS(def->status);
  for (auto&& item : def->attrs) {
    if (item.default_value.attr_type != AttrValue::kNone) {
      // Do not write the exist key
      attr->insert({item.name, item.default_value});
    }
    auto iter = attr->find(item.name);
    XDL_CHECK_COND(iter != attr->end(),
                   Status::ArgumentError("Attr not found " + item.name));
    XDL_CHECK_COND(iter->second.attr_type == item.type,
                   Status::ArgumentError("Attr type Error " + item.name));
  }
  return Status::Ok();
}

Status OpDefine::GetDef(const std::string& op, OpDefineItem** def) {
  auto iter = definitions_.find(op);
  XDL_CHECK_COND(iter != definitions_.end(),
                 Status::ArgumentError("op Not Found " + op));
  *def = iter->second.get();
  return Status::Ok();
}

Status OpDefine::ParseRefDataType(
    const OpDefineItem::RefDataType& ref,
    const std::unordered_map<std::string, AttrValue>& attr,
    DataType* result) {
  if (ref.attr != "") {
    auto&& iter = attr.find(ref.attr);
    XDL_CHECK_COND(iter != attr.end(),
                   Status::ArgumentError("Attr Not Found " + ref.attr));
    auto& value = iter->second;
    XDL_CHECK_COND(value.attr_type == AttrValue::kDataType,
                   Status::ArgumentError("Attr Type Error " + ref.attr));
    *result = value.type;
    return Status::Ok();
  } else {
    *result = ref.raw;
    return Status::Ok();
  }
}

Status OpDefine::ParseRefInt64(
    const OpDefineItem::RefInt64& ref,
    const std::unordered_map<std::string, AttrValue>& attr,
    int* result) {
  if (ref.attr != "") {
    auto&& iter = attr.find(ref.attr);
    XDL_CHECK_COND(iter != attr.end(),
                   Status::ArgumentError("Attr Not Found " + ref.attr));
    auto& value = iter->second;
    XDL_CHECK_COND(value.attr_type == AttrValue::kInt,
                   Status::ArgumentError("Attr Type Error " + ref.attr));
    *result = value.i;
    return Status::Ok();
  } else {
    *result = ref.raw;
    return Status::Ok();
  }
}

Status OpDefine::ParseDataTypeList(
    const std::string& name,
    const std::unordered_map<std::string, AttrValue>& attr,
    std::vector<DataType>* result) {
  auto&& iter = attr.find(name);
  XDL_CHECK_COND(iter != attr.end(),
                 Status::ArgumentError("Attr Not Found " + name));
  auto& value = iter->second;
  XDL_CHECK_COND(value.attr_type == AttrValue::kDataTypeList,
                 Status::ArgumentError("Attr Type Error " + name));
  *result = value.type_list;
  return Status::Ok();
}

OpDefineHelper::OpDefineHelper(const std::string& name)
  : name_(name), internal_(new OpDefineItem(name)) {}

OpDefineHelper& OpDefineHelper::Input(
    const std::string& name,
    const DataTypeWrapper& type) {
  internal_->inputs.push_back(OpDefineItem::Input{
      .name = name,
      .type = OpDefineItem::DType{.repeated = OpDefineItem::kNoRepeat,
                                  .type = type.Internal()}
  });
  return *this;
}

OpDefineHelper& OpDefineHelper::InputList(
    const std::string& name,
    const DataTypeWrapper& type,
    const IntWrapper& size) {
  internal_->inputs.push_back(OpDefineItem::Input{
      .name = name,
      .type = OpDefineItem::DType{.repeated = OpDefineItem::kTypeAndSize,
                                  .type = type.Internal(),
                                  .size = size.Internal()}
  });
  return *this;
}

OpDefineHelper& OpDefineHelper::InputListV2(
    const std::string& name,
    const std::string& type_list) {
  OpDefineItem::Input input;
  input.name = name;
  input.type.repeated = OpDefineItem::kTypeList;
  input.type.type_list = type_list;
  internal_->inputs.push_back(input);
  return *this;
}

OpDefineHelper& OpDefineHelper::Output(
    const std::string& name,
    const DataTypeWrapper& type) {
  internal_->outputs.push_back(OpDefineItem::Output{
      .name = name,
      .type = OpDefineItem::DType{.repeated = OpDefineItem::kNoRepeat,
                                  .type = type.Internal()}
  });
  return *this;
}

OpDefineHelper& OpDefineHelper::OutputList(
    const std::string& name,
    const DataTypeWrapper& type,
    const IntWrapper& size) {
  internal_->outputs.push_back(OpDefineItem::Output{
      .name = name,
      .type = OpDefineItem::DType{.repeated = OpDefineItem::kTypeAndSize,
                                  .type = type.Internal(),
                                  .size = size.Internal()}
  });
  return *this;
}

OpDefineHelper& OpDefineHelper::OutputListV2(
    const std::string& name,
    const std::string& type_list) {
  OpDefineItem::Output output;
  output.name = name;
  output.type.repeated = OpDefineItem::kTypeList;
  output.type.type_list = type_list;
  internal_->outputs.push_back(output);
  return *this;
}

OpDefineHelper& OpDefineHelper::Attr(
    const std::string& name,
    AttrValue::Type type) {
  internal_->attrs.push_back(OpDefineItem::Attr{
      .name = name,
      .type = type,
      .default_value = AttrValue{.attr_type = AttrValue::kNone}
  });
  return *this;
}

OpDefineHelper& OpDefineHelper::Attr(
    const std::string& name,
    AttrValue::Type type,
    const AttrWrapper& default_value) {
  internal_->attrs.push_back(OpDefineItem::Attr{
      .name = name,
      .type = type,
      .default_value = default_value.Internal()
  });
  return *this;
}

OpDefineHelper& OpDefineHelper::Tag(const std::string& tag) {
  internal_->tags.push_back(tag);
  return *this;
}

}  // namespace xdl


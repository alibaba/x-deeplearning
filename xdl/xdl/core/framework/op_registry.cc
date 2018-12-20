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

#include "xdl/core/framework/op_registry.h"

namespace xdl {

namespace {

bool AttrMatch(const AttrValue& pattern, const AttrValue& attr) {
  if (pattern.attr_type != attr.attr_type) {
    return false;
  }
  switch (pattern.attr_type) {
  case AttrValue::kNone:
    return true;
  case AttrValue::kString:
    return pattern.s == attr.s;
  case AttrValue::kInt:
    return pattern.i == attr.i;
  case AttrValue::kFloat:
    return pattern.f == attr.f;
  case AttrValue::kBool:
    return pattern.b == attr.b;
  case AttrValue::kDataType:
    return pattern.type == attr.type;
  case AttrValue::kTensorShape:
    return pattern.shape == attr.shape;
  case AttrValue::kDataTypeList:
    if (pattern.type_list.size() != attr.type_list.size()) {
      return false;
    }
    for (size_t i = 0; i < pattern.type_list.size(); i++) {
      if (pattern.type_list[i] != attr.type_list[i]) {
        return false;
      }
    }
    return true;
  default:
    XDL_CHECK(false) << "AttrMatch reach a unknown attr";
  }
}

bool AttrMapMatch(const std::unordered_map<std::string, AttrValue>& pattern,
                  const std::unordered_map<std::string, AttrValue>& attr) {
  for (auto&& item : pattern) {
    auto iter = attr.find(item.first);
    if (iter == attr.end()) {
      return false;
    }
    if (!AttrMatch(item.second, iter->second)) {
      return false;
    }
  }
  return true;
}

}  // namespace

OpRegistryItemList::~OpRegistryItemList() {
  for (auto item : items_) {
    delete item;
  }
}

void OpRegistryItemList::RegisterOp(OpRegistryItem* item) {
  items_.insert(item);
}

OpRegistryItem* OpRegistryItemList::FindItem(
    const std::string& device,
    const std::unordered_map<std::string, AttrValue>& attr,
    bool ignore_device) {
  for (auto&& item : items_) {
    if (!ignore_device) {
      if (item->device != "" && item->device != device) {
        continue;
      }
    }

    if (!AttrMapMatch(item->attr, attr)) {
      continue;
    }

    return item;
  }
  return nullptr;
}

Status OpRegistry::CreateKernel(const NodeDef& node, const std::string& device,
                                OpKernelBase** kernel) {
  auto iter = items_.find(node.op);
  XDL_CHECK_COND(iter != items_.end(),
                 Status::ArgumentError("Op Not Found " + node.op));
  OpRegistryItem* ret = iter->second.FindItem(device, node.attr);
  if (ret == nullptr) {
    ret = iter->second.FindItem(device, node.attr, true);
  }
  
  XDL_CHECK_COND(ret != nullptr,
                 Status::ArgumentError("Op Not Found " + node.op));
  *kernel = ret->factory();
  return Status::Ok();
}

void OpRegistry::RegisterOp(const std::string& name, OpRegistryItem* item) {
  items_[name].RegisterOp(item);
}

OpRegisterHelper::OpRegisterHelper(const std::string& name)
  : name_(name), item_(new OpRegistryItem) {
    item_->priority = OpRegistry::kDefaultPriority;
}

OpRegisterHelper& OpRegisterHelper::Factory(
    const std::function<OpKernelBase*()>& factory) {
  item_->factory = factory;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::Priority(int priority) {
  item_->priority = priority;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::Device(const std::string& device) {
  item_->device = device;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::AttrInt(const std::string& name,
                                            int64_t i) {
  AttrValue& attr = item_->attr[name];
  attr.attr_type = AttrValue::kInt;
  attr.i = i;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::AttrFloat(const std::string& name,
                                              float f) {
  AttrValue& attr = item_->attr[name];
  attr.attr_type = AttrValue::kFloat;
  attr.f = f;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::AttrString(const std::string& name,
                                               const std::string& str) {
  AttrValue& attr = item_->attr[name];
  attr.attr_type = AttrValue::kString;
  attr.s = str;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::AttrTensorShape(const std::string& name,
                                                    const TensorShape& shape) {
  AttrValue& attr = item_->attr[name];
  attr.attr_type = AttrValue::kTensorShape;
  attr.shape = shape;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::AttrDataType(const std::string& name,
                                                 DataType type) {
  AttrValue& attr = item_->attr[name];
  attr.attr_type = AttrValue::kDataType;
  attr.type = type;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::AttrDataTypeList(
    const std::string& name,
    const std::vector<DataType>& type_list) {
  AttrValue& attr = item_->attr[name];
  attr.attr_type = AttrValue::kDataTypeList;
  attr.type_list = type_list;
  return *this;
}

OpRegisterHelper& OpRegisterHelper::AttrDataTypeList(
    const std::string& name,
    const std::initializer_list<DataType>& type_list) {
  AttrValue& attr = item_->attr[name];
  attr.attr_type = AttrValue::kDataTypeList;
  attr.type_list = std::vector<DataType>(type_list);
  return *this;
}

}  // namespace xdl


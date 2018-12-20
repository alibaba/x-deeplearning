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

#ifndef XDL_CORE_FRAMEWORK_OP_REGISTRY_H_
#define XDL_CORE_FRAMEWORK_OP_REGISTRY_H_

#include <set>
#include <string>
#include <unordered_map>
#include <functional>

#include "xdl/core/lib/singleton.h"
#include "xdl/core/framework/tensor_shape.h"
#include "xdl/core/framework/types.h"
#include "xdl/core/framework/graph_def.h"
#include "xdl/core/framework/op_kernel.h"

namespace xdl {

struct OpRegistryItem {
  std::string device;
  std::unordered_map<std::string, AttrValue> attr;
  std::function<OpKernelBase*()> factory;
  int priority;
};

class OpRegistryItemList {
 public:
  ~OpRegistryItemList();
  void RegisterOp(OpRegistryItem* item);
  OpRegistryItem* FindItem(
      const std::string& device,
      const std::unordered_map<std::string, AttrValue>& attr,
      bool ignore_device = false);
 private:
  class OpRegistryItemCompare {
   public:
    bool operator()(OpRegistryItem* lhs, OpRegistryItem* rhs) {
      if (lhs->priority != rhs->priority) {
        return lhs->priority > rhs->priority;
      }
      return lhs > rhs;
    }
  };
  std::set<OpRegistryItem*, OpRegistryItemCompare> items_;
};

class OpRegistry : public Singleton<OpRegistry> {
 public:
  static constexpr int kDefaultPriority = 0;
  Status CreateKernel(const NodeDef& node, const std::string& device,
                      OpKernelBase** kernel);
  void RegisterOp(const std::string& name, OpRegistryItem* item);
 private:
  std::unordered_map<std::string, OpRegistryItemList> items_;
};

class OpRegisterHelper {
 public:
  explicit OpRegisterHelper(const std::string& name);
  template <typename T>
  OpRegisterHelper& OpKernel();
  OpRegisterHelper& Factory(const std::function<OpKernelBase*()>& factory);
  OpRegisterHelper& Priority(int priority);
  OpRegisterHelper& Device(const std::string& device);
  OpRegisterHelper& AttrInt(const std::string& name, int64_t i);
  OpRegisterHelper& AttrFloat(const std::string& name, float f);
  OpRegisterHelper& AttrString(const std::string& name,
                               const std::string& str);
  OpRegisterHelper& AttrTensorShape(const std::string& name,
                                    const TensorShape& shape);
  OpRegisterHelper& AttrDataType(const std::string& name, DataType type);
  template <typename T>
  OpRegisterHelper& AttrDataType(const std::string& name);
  OpRegisterHelper& AttrDataTypeList(
      const std::string& name,
      const std::initializer_list<DataType>& type_list);
  OpRegisterHelper& AttrDataTypeList(
      const std::string& name,
      const std::vector<DataType>& type_list);
  std::string Name() const { return name_; }
  OpRegistryItem* Internal() const { return item_; }
 private:
  std::string name_;
  OpRegistryItem* item_;
};

class OpRegisterHelperReceiver {
 public:
  OpRegisterHelperReceiver(
      const OpRegisterHelper& helper) {  // NOLINT(runtime/explicit)
    OpRegistry::Get()->RegisterOp(helper.Name(), helper.Internal());
  }
};

template <typename T>
OpRegisterHelper& OpRegisterHelper::OpKernel() {
  return Factory([]()->OpKernelBase* { return new T; });
}

template <typename T>
OpRegisterHelper& OpRegisterHelper::AttrDataType(const std::string& name) {
  return AttrDataType(name, DataTypeToEnum<T>::v());
}

}  // namespace xdl

#define XDL_REGISTER_SINGLE_ARGS(...) __VA_ARGS__
#define XDL_REGISTER_KERNEL(name, ...)                        \
  XDL_REGISTER_KERNEL_CORE(name).OpKernel<__VA_ARGS__>()
#define XDL_REGISTER_KERNEL_CORE(name)                        \
  XDL_REGISTER_KERNEL_UNIQ_HELPER(__COUNTER__, name)
#define XDL_REGISTER_KERNEL_UNIQ_HELPER(ctr, name)            \
  XDL_REGISTER_KERNEL_UNIQ(ctr, name)
#define XDL_REGISTER_KERNEL_UNIQ(ctr, name)                   \
  static ::xdl::OpRegisterHelperReceiver __register_op__##ctr \
  __attribute__((unused)) = ::xdl::OpRegisterHelper(#name)

#endif  // XDL_CORE_FRAMEWORK_OP_REGISTRY_H_


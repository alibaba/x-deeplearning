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

#ifndef XDL_CORE_FRAMEWORK_OP_DEFINE_H_
#define XDL_CORE_FRAMEWORK_OP_DEFINE_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <type_traits>
#include <initializer_list>

#include "xdl/core/lib/singleton.h"
#include "xdl/core/framework/op_kernel.h"

namespace xdl {

struct OpDefineItem {
  enum RepeatType {
    kNoRepeat = 0,
    kTypeAndSize = 1,
    kTypeList = 2
  };
  struct RefDataType {
    std::string attr;
    DataType raw;
  };
  struct RefInt64 {
    std::string attr;
    int64_t raw;
  };
  struct DType {
    RepeatType repeated;
    RefDataType type;
    RefInt64 size;
    std::string type_list;
  };
  struct Input {
    std::string name;
    DType type;
  };
  struct Output {
    std::string name;
    DType type;
  };
  struct Attr {
    std::string name;
    AttrValue::Type type;
    AttrValue default_value;
  };
  Status status;
  std::string name;
  std::vector<Input> inputs;
  std::vector<Output> outputs;
  std::vector<Attr> attrs;
  std::vector<std::string> tags;
  OpDefineItem(const std::string& name_) : name(name_) {}
};

class OpDefine : public Singleton<OpDefine> {
 public:
  void DefineOp(const std::string& name, OpDefineItem* item);
  std::unordered_map<std::string, OpDefineItem*> GetDefinitions();
  Status Validate(const std::string& op,
                  const std::unordered_map<std::string, AttrValue>& attr);
  Status PrepareOpKernelContext(
      const std::string& op,
      std::unordered_map<std::string, AttrValue>* attr,
      OpKernelContextArg* arg);
  Status GetTag(const std::string& op, std::vector<std::string>* tag);

 private:
  Status ValidateDef(OpDefineItem* def);
  Status GetDef(const std::string& op, OpDefineItem** def);
  Status PrepareAttr(OpDefineItem* def,
                     std::unordered_map<std::string, AttrValue>* attr);
  Status ParseRefDataType(
      const OpDefineItem::RefDataType& ref,
      const std::unordered_map<std::string, AttrValue>& attr,
      DataType* result);
  Status ParseRefInt64(
      const OpDefineItem::RefInt64& ref,
      const std::unordered_map<std::string, AttrValue>& attr,
      int* result);
  Status ParseDataTypeList(
      const std::string& name,
      const std::unordered_map<std::string, AttrValue>& attr,
      std::vector<DataType>* result);

  std::unordered_map<std::string, std::unique_ptr<OpDefineItem>> definitions_;
};

class OpDefineHelper {
 public:
  explicit OpDefineHelper(const std::string& name);
  class DataTypeWrapper {
   public:
    DataTypeWrapper(const std::string& attr)  // NOLINT(runtime/explicit)
      : internal_(OpDefineItem::RefDataType{.attr = attr}) {}
    DataTypeWrapper(const char* attr)  // NOLINT(runtime/explicit)
      : internal_(OpDefineItem::RefDataType{.attr = attr}) {}
    DataTypeWrapper(DataType raw)  // NOLINT(runtime/explicit)
      : internal_(OpDefineItem::RefDataType{.attr = "", .raw = raw}) {}
    OpDefineItem::RefDataType Internal() const { return internal_; }
   private:
    OpDefineItem::RefDataType internal_;
  };
  class IntWrapper {
   public:
    IntWrapper(const std::string& attr)  // NOLINT(runtime/explicit)
      : internal_(OpDefineItem::RefInt64{.attr = attr}) {}
    IntWrapper(const char* attr)  // NOLINT(runtime/explicit)
      : internal_(OpDefineItem::RefInt64{.attr = attr}) {}
    template<
      class T,
      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    IntWrapper(T raw)  // NOLINT(runtime/explicit)
      : internal_(OpDefineItem::RefInt64{.attr = "", .raw = raw}) {}
    OpDefineItem::RefInt64 Internal() const { return internal_; }
   private:
    OpDefineItem::RefInt64 internal_;
  };
  class AttrWrapper {
   public:
    AttrWrapper(const std::string& raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kString;
      internal_.s = raw;
    }
    AttrWrapper(const char* raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kString;
      internal_.s = raw;
    }
    template<
      class T,
      typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    AttrWrapper(T raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kInt;
      internal_.i = raw;
    }
    template<
      class T,
      typename std::enable_if<std::is_floating_point<T>::value, int>::type>
    AttrWrapper(T raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kFloat;
      internal_.f = raw;
    }
    AttrWrapper(bool raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kBool;
      internal_.b = raw;
    }
    AttrWrapper(DataType raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kDataType;
      internal_.type = raw;
    }
    AttrWrapper(const TensorShape& raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kTensorShape;
      internal_.shape = raw;
    }
    AttrWrapper(const std::vector<DataType>& raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kDataTypeList;
      internal_.type_list = raw;
    }
    AttrWrapper(const std::initializer_list<DataType>& raw) {  // NOLINT(runtime/explicit)
      internal_.attr_type = AttrValue::kDataTypeList;
      internal_.type_list = std::vector<DataType>(raw);
    }

    AttrValue Internal() const { return internal_; }

   private:
    AttrValue internal_;
  };
  OpDefineHelper& Input(
      const std::string& name,
      const DataTypeWrapper& type);
  OpDefineHelper& InputList(
      const std::string& name,
      const DataTypeWrapper& type,
      const IntWrapper& size);
  OpDefineHelper& InputListV2(
      const std::string& name,
      const std::string& type_list);
  OpDefineHelper& Output(
      const std::string& name,
      const DataTypeWrapper& type);
  OpDefineHelper& OutputList(
      const std::string& name,
      const DataTypeWrapper& type,
      const IntWrapper& size);
  OpDefineHelper& OutputListV2(
      const std::string& name,
      const std::string& type_list);
  OpDefineHelper& Attr(
      const std::string& name,
      AttrValue::Type type);
  OpDefineHelper& Attr(
      const std::string& name,
      AttrValue::Type type,
      const AttrWrapper& default_value);
  OpDefineHelper& Tag(const std::string& tag);

  std::string Name() const { return name_; }
  OpDefineItem* Internal() const { return internal_; }

 private:
  std::string name_;
  OpDefineItem* internal_;
};

class OpDefineHelperReceiver {
 public:
  OpDefineHelperReceiver(
      const OpDefineHelper& helper) {  // NOLINT(runtime/explicit)
    OpDefine::Get()->DefineOp(helper.Name(), helper.Internal());
  }
};

#define XDL_DEFINE_OP(name)                               \
  XDL_DEFINE_OP_UNIQ_HELPER(__COUNTER__, name)
#define XDL_DEFINE_OP_UNIQ_HELPER(ctr, name)              \
  XDL_DEFINE_OP_UNIQ(ctr, name)
#define XDL_DEFINE_OP_UNIQ(ctr, name)                     \
  static ::xdl::OpDefineHelperReceiver __define_op__##ctr \
  __attribute__((unused)) = ::xdl::OpDefineHelper(#name)

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_OP_DEFINE_H_


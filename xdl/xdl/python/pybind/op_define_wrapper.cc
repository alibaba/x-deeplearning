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

#include "xdl/python/pybind/op_define_wrapper.h"

#include "xdl/core/framework/op_define.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

PYBIND11_MAKE_OPAQUE(std::vector<xdl::OpDefineItem::Input>);
PYBIND11_MAKE_OPAQUE(std::vector<xdl::OpDefineItem::Output>);
PYBIND11_MAKE_OPAQUE(std::vector<xdl::OpDefineItem::Attr>);
PYBIND11_MAKE_OPAQUE(std::vector<xdl::OpDefineItem>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);

namespace xdl {
namespace python_lib {

std::vector<OpDefineItem>
    OpDefineRegistryWrapper::GetLatestOpDefineItem() {
  std::vector<OpDefineItem> ret;
  std::unordered_map<std::string, OpDefineItem*> defs =
      OpDefine::Get()->GetDefinitions();
  for (auto&& def_item : defs) {
    if (!visited_op_.insert(def_item.first).second) {
      continue;
    }
    ret.push_back(OpDefineItem(*def_item.second));
  }
  return ret;
}

std::vector<OpDefineItem> GetLatestOpDefineItem() {
  return OpDefineRegistryWrapper::Get()->GetLatestOpDefineItem();
}

void OpDefinePybind(pybind11::module& m) {
  pybind11::class_<OpDefineItem> op_define_item(m, "OpDefineItem");
  op_define_item
    .def_readwrite("status", &OpDefineItem::status)
    .def_readwrite("name", &OpDefineItem::name)
    .def_readwrite("inputs", &OpDefineItem::inputs)
    .def_readwrite("outputs", &OpDefineItem::outputs)
    .def_readwrite("attrs", &OpDefineItem::attrs)
    .def_readwrite("tags", &OpDefineItem::tags);

  pybind11::enum_<OpDefineItem::RepeatType>(op_define_item, "RepeatType")
    .value("no_repeat", OpDefineItem::RepeatType::kNoRepeat)
    .value("type_and_size", OpDefineItem::RepeatType::kTypeAndSize)
    .value("type_list", OpDefineItem::RepeatType::kTypeList);

  pybind11::class_<OpDefineItem::RefDataType>(op_define_item, "RefDataType")
    .def_readwrite("attr", &OpDefineItem::RefDataType::attr)
    .def_readwrite("raw", &OpDefineItem::RefDataType::raw);

  pybind11::class_<OpDefineItem::RefInt64>(op_define_item, "RefInt64")
    .def_readwrite("attr", &OpDefineItem::RefInt64::attr)
    .def_readwrite("raw", &OpDefineItem::RefInt64::raw);

  pybind11::class_<OpDefineItem::DType>(op_define_item, "DType")
    .def_readwrite("repeated", &OpDefineItem::DType::repeated)
    .def_readwrite("type", &OpDefineItem::DType::type)
    .def_readwrite("size", &OpDefineItem::DType::size)
    .def_readwrite("type_list", &OpDefineItem::DType::type_list);

  pybind11::class_<OpDefineItem::Input>(op_define_item, "Input")
    .def_readwrite("name", &OpDefineItem::Input::name)
    .def_readwrite("type", &OpDefineItem::Input::type);

  pybind11::class_<OpDefineItem::Output>(op_define_item, "Output")
    .def_readwrite("name", &OpDefineItem::Output::name)
    .def_readwrite("type", &OpDefineItem::Output::type);

  pybind11::class_<OpDefineItem::Attr>(op_define_item, "Attr")
    .def_readwrite("name", &OpDefineItem::Attr::name)
    .def_readwrite("type", &OpDefineItem::Attr::type)
    .def_readwrite("default_value", &OpDefineItem::Attr::default_value);

  pybind11::bind_vector<std::vector<OpDefineItem::Input>>(
      op_define_item, "InputVector");

  pybind11::bind_vector<std::vector<OpDefineItem::Output>>(
      op_define_item, "OutputVector");

  pybind11::bind_vector<std::vector<OpDefineItem::Attr>>(
      op_define_item, "AttrVector");

  pybind11::bind_vector<std::vector<OpDefineItem>>(
      m, "OpDefineItemVector");

  m.def("GetLatestOpDefineItem", &GetLatestOpDefineItem,
      "Get latest registered ops");
}

}  // namespace python_lib
}  // namespace xdl


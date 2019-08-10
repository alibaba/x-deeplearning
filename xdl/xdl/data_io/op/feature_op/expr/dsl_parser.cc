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


#include "xdl/data_io/op/feature_op/expr/dsl_parser.h"

#include <xdl/core/utils/logging.h>
#include <stdio.h>

#include "xdl/data_io/op/feature_op/string_util.h"

namespace xdl {
namespace io {

const std::string DslParser::item_separator_ = ";";
const std::string DslParser::kv_separator_ = "=";

void DslParser::Parse(const std::string &dsl) {
  std::vector<std::string> conf_items;
  StringUtil::Split(dsl, conf_items, item_separator_);
  if (conf_items.size() == 0)  return;
  DslUnit dsl_unit;
  for (std::string &conf_item : conf_items) {
    std::vector<std::string> conf_kvs;
    StringUtil::Split(conf_item, conf_kvs, kv_separator_);
    XDL_CHECK(conf_kvs.size() == 2);
    const std::string &key = StringUtil::Trim(conf_kvs[0]);
    const std::string &value = StringUtil::Trim(conf_kvs[1]);
    if (key == "Name") {
      dsl_unit.name = std::move(value);
    } else if (key == "Expr") {
      dsl_unit.expr = std::move(value);
    } else if (key == "Type") {
      if (value == "Numeric")  dsl_unit.type = DslType::kDslNumeric;
      else if (value == "KV")  dsl_unit.type = DslType::kDslKv;
      else if (value == "ID")  dsl_unit.type = DslType::kDslId;
      else  XDL_CHECK(false);
    } else {
      XDL_CHECK(false);
    }
  }
  if (dsl_unit.IsEmpty()) {
    printf("illegal dsl: %s\n", dsl.c_str());
    printf("  parse result: name=\"%s\", expr=\"%s\", type=%d\n",
           dsl_unit.name.c_str(), dsl_unit.expr.c_str(), dsl_unit.type);
    XDL_CHECK(false);
  }
  XDL_CHECK(dsl_unit_map_.find(dsl_unit.name) == dsl_unit_map_.end());
  dsl_unit_map_.insert(std::make_pair(dsl_unit.name, dsl_unit));
}

}  // namespace io
}  // namespace xdl

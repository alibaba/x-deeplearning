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

#include "gtest/gtest.h"

#include <unordered_map>

#include "xdl/data_io/op/feature_op/expr/dsl_parser.h"
#include "xdl/data_io/op/feature_op/expr/dsl_unit.h"

using xdl::io::DslUnit;
using xdl::io::DslType;
using xdl::io::DslUnitMap;

void CheckDslUnit(const DslUnitMap &dsl_unit_map,
                  const std::string &name,
                  const std::string &expr,
                  DslType dsl_type) {
  const auto &iter = dsl_unit_map.find(name);
  EXPECT_TRUE(iter != dsl_unit_map.end());
}

TEST(DslParserTest, Default) {
  using xdl::io::DslParser;

  const size_t size = 4;
  const std::string name_arr[size] = { "ad_cate_id",
                                       "nick_cate_buy_14",
                                       "ad_cate_pv_14_match_id",
                                       "ad_cate_pv_14_match_sum" };
  const std::string expr_arr[size] = { "hash(ad.ad_cate_id)",
                                       "log(hash( nick.nick_cate_buy_14)) )",
                                       "match(ad_cate_id, nick_cate_pv_14))",
                                       "sum(value(match (ad_cate_id,nick_cate_pv_14)))" };
  const std::string type_arr[size] = { "ID",
                                       "KV",
                                       "KV",
                                       "Numeric" };
  const DslType dsl_type_arr[size] = { DslType::kDslId,
                                       DslType::kDslKv,
                                       DslType::kDslKv,
                                       DslType::kDslNumeric };

  DslParser *dsl_parser = DslParser::Get();
  dsl_parser->Parse("Name =" + name_arr[0] + ";Expr= " + expr_arr[0] + ";Type=" + type_arr[0] + ";");
  dsl_parser->Parse(" Name= " + name_arr[1] + ";Expr = " + expr_arr[1] + " ; Type= " + type_arr[1]);
  dsl_parser->Parse("Name=" + name_arr[2] + "　; Expr=" + expr_arr[2] + ";Type=" + type_arr[2] + " ;");
  dsl_parser->Parse("Name=" + name_arr[3] + ";Expr=" + expr_arr[3] + ";Type=" + type_arr[3] + "");
  const DslUnitMap &dsl_unit_map = dsl_parser->dsl_unit_map();

  EXPECT_EQ(dsl_unit_map.size(), size);
  for (size_t i = 0; i < size; ++i) {
    CheckDslUnit(dsl_unit_map, name_arr[i], expr_arr[i], dsl_type_arr[i]);
  }
}

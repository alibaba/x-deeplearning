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

#include "xdl/data_io/parser/parse_txt.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "gtest/gtest.h"

namespace xdl {
namespace io {

TEST(ParseTxtTest, TestTokenize) {
  std::vector<std::string>vals = {"aaa", "bbbbb", "c", "dd"};
  std::string str;
  for (int i = 0; i < vals.size(); ++i) {
    str += vals[i];
    if (i != vals.size() - 1) {
      str += kVAL;
    }
  }
  const char *ptrs[MAX_NUM_VAL];
  int lens[MAX_NUM_VAL];
  int n = ParseTxt::Tokenize(str.c_str(), str.size(), kVAL, MAX_NUM_VAL, [&ptrs, &lens](const char *s, size_t l, int i) mutable{
                                //std::cout << "[" << i << "] " << std::string(s, l) << std::endl;
                                ptrs[i] = s;
                                lens[i] = l;
                              });
  EXPECT_EQ(vals.size(), n);
  for (int i = 0; i < n; ++i) {
    //std::cout << "[" << i << "] " << ptrs[i] << lens[i] << std::endl;
    
    EXPECT_EQ(vals[i].size(), lens[i]);
    //EXPECT_EQ(0, strncmp(vals[i].c_str(), ptrs[i], lens[i]));
  }
}

// Disable it due to OnSparse OnDense is inline
/*
TEST(ParseTxtTest, TestOnSparse) {
  FeatureValue fv;
  const char *str = "111:222";
  EXPECT_TRUE(ParseTxt::OnSparse(&fv, str, strlen(str)));
  EXPECT_EQ(111, fv.key());
  EXPECT_EQ(222, fv.value());
}

TEST(ParseTxtTest, TestOnDense) {
  FeatureValue fv;
  const char *str = "111";
  EXPECT_TRUE(ParseTxt::OnDense(&fv, str, strlen(str)));
  EXPECT_EQ(111, fv.value());
}
*/

TEST(ParseTxtTest, TestRun) {
  typedef std::map<std::string, std::vector<std::pair<unsigned, float>>> SMap;
  typedef std::map<std::string, std::vector<float>> DMap;

  SMap sparse_features;
  DMap dense_features;

  sparse_features.insert(SMap::value_type("upv14", {{101,0.1}, {102,0.2}, {101,0.3}}));
  sparse_features.insert(SMap::value_type("ufav3", {{201,0.1}}));
  
  dense_features.insert(DMap::value_type("a", {0.01,0.02,0.03}));
  dense_features.insert(DMap::value_type("s", {0.04}));

  std::vector<float> labels = {1.0, 0.0};

  /// sample key| group key| kvs| dense| label| time
  std::string str = "skey|gkey|";
  for (auto iter = sparse_features.cbegin();
       iter != sparse_features.cend(); ++iter) {

    if (iter != sparse_features.cbegin()) {
      str += ";";
    }

    const auto &f = *iter;
    str += f.first + "@";
    for (size_t i = 0; i < f.second.size(); ++i) {
      const auto fv = f.second[i];
      str += std::to_string(fv.first)+":"+std::to_string(fv.second);
      if (i < f.second.size() - 1) {
        str += ",";
      }
    }
  }

  str += "|";
  for (auto iter = dense_features.cbegin();
       iter != dense_features.cend(); ++iter) {

    if (iter != dense_features.cbegin()) {
      str += ";";
    }

    const auto &f = *iter;
    str += f.first + "@";
    for (size_t j = 0; j < f.second.size(); ++j) {
      const auto fv = f.second[j];
      str += std::to_string(fv);
      if (j < f.second.size() - 1) {
        str += ",";
      }
    }
  }

  str += "|";
  for (auto iter = labels.cbegin();
       iter != labels.cend(); ++iter) {
    if (iter != labels.cbegin()) {
      str += ",";
    }
    float v = *iter;
    str += std::to_string(v);
  }

  str += "|150000000";

  std::cout << str << std::endl;

  Schema schema;
  ParseTxt p(&schema);
  p.InitMeta("");
  auto sgroup = p.Run(str.c_str(), str.size());
  EXPECT_EQ(nullptr, sgroup);
  sgroup = p.Run(str.c_str(), 0);
  EXPECT_NE(nullptr, sgroup);
  auto sg = sgroup->Get();
  std::cout << sg << std::endl << sg->ShortDebugString() << std::endl;

  EXPECT_STREQ("skey", sg->sample_ids(0).c_str());

  auto fl = sg->feature_tables(0).feature_lines(0);

  EXPECT_EQ(sparse_features.size() + dense_features.size(), fl.features_size());

  for (int k = 0; k < fl.features_size(); ++k) {
    auto f = fl.features(k);
    if (k < sparse_features.size()) {
      EXPECT_EQ(kSparse, f.type());
      auto it = sparse_features.find(f.name());
      EXPECT_NE(sparse_features.end(), it);
      EXPECT_EQ(it->second.size(), f.values_size());
      for (int i = 0; i < f.values_size(); ++i) {
        auto v = f.values(i);
        auto v_ = it->second[i];
        EXPECT_EQ(v_.first, v.key());
        EXPECT_EQ(v_.second, v.value());
      }
    } else {
      EXPECT_EQ(kDense, f.type());
      auto it = dense_features.find(f.name());
      EXPECT_NE(dense_features.end(), it);
      EXPECT_EQ(1, f.values_size());
      EXPECT_FALSE(f.values(0).has_key());
      EXPECT_EQ(it->second.size(), f.values(0).vector_size());
      for (int i = 0; i < f.values(0).vector_size(); ++i) {
        auto v = f.values(0).vector(i);
        auto v_ = it->second[i];
        EXPECT_EQ(v_, v);
      }
    }
  }

}

}  // namespace io
}  // namespace xdl


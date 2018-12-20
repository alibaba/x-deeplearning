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

#pragma once

#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

#include "xdl/data_io/op/feature_op/expr/expr_node.h"
#include "xdl/proto/sample.pb.h"

class OpTestTool {
 public:
  static int Rand(int range) {
    srand((unsigned) time(nullptr));
    return rand() / (RAND_MAX / range);
  }

  static float Rand(float range) {
    srand((unsigned) time(nullptr));
    return rand() / (RAND_MAX / range);
  }

  static void Rand(char str[], int size, int range) {
    str[--size] = 0;
    const char c = 'A' + Rand(range);
    for (int i = 0; i < size; ++i) {
      str[i] = c;
    }
  }

  static void Rand(std::string &str, int size_range, int range) {
    if (!str.empty())  str.clear();
    int size = Rand(size_range);
    str.resize(size);
    for (int i = 0; i < size; ++i) {
      str[i] = 'A' + Rand(range);
    }
  }

  static double GetTime() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
  }

  static void Transfer(const std::vector<const xdl::io::Feature *> &features,
                       xdl::io::Feature &result_feature,
                       std::vector<xdl::io::ExprNode> &nodes,
                       std::vector<const xdl::io::ExprNode *> &source_nodes,
                       xdl::io::ExprNode &result_node) {
    for (const xdl::io::Feature *feature : features) {
      xdl::io::ExprNode node;
      node.type = xdl::io::FeaOpType::kSourceFeatureOp;
      xdl::io::Feature *f = new xdl::io::Feature();
      f->CopyFrom(*feature);
      node.result = f;
      node.InitInternal();
      nodes.push_back(std::move(node));
    }
    for (const xdl::io::ExprNode &node : nodes) {
      source_nodes.push_back(&node);
    }
    result_node.type = xdl::io::FeaOpType::kMultiFeatureOp;
    result_node.output = true;
    result_node.result = &result_feature;
    result_node.InitInternal();
  }
};

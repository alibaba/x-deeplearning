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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#include "gtest/gtest.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tdm/tdm_op.h"

namespace tdm {

static int Rand(int range) {
  return rand() / (RAND_MAX / range);
}

static float Rand(float range) {
  return rand() / (RAND_MAX / range);
}

using xdl::io::SampleGroup;
using xdl::io::FeatureTable;
using xdl::io::FeatureLine;
using xdl::io::Feature;
using xdl::io::FeatureValue;
using xdl::io::FeatureType;
using xdl::io::Label;

static void PrintTree(DistTree &dist_tree) {
  printf("==== PrintTree: max_level=%d\n", dist_tree.max_level());
  for (int level = 0; level < dist_tree.max_level(); ++level) {
    printf("level[%d] LevelNeighborCount=%ld\n",
           level,
           static_cast<size_t>(pow(dist_tree.branch(), level)));
    auto level_itr = dist_tree.LevelIterator(level);
    auto level_end = dist_tree.LevelEnd(level);
    for (int sp = 0; sp < 3 && level_itr != level_end; ++sp, ++level_itr) {
      Node node;
      node.ParseFromString(level_itr->value);
      if (node.is_leaf()) printf("level[%d] leaf.id=%ld\n", level, node.id());
    }
  }
  printf("==== \n");
}

static void BuildIdSets(DistTree &dist_tree,
                        std::vector<std::set<int64_t>> &id_sets) {
  for (int level = 0; level < dist_tree.max_level(); ++level) {
    auto level_itr = dist_tree.LevelIterator(level);
    auto level_end = dist_tree.LevelEnd(level);
    for (; level_itr != level_end; ++level_itr) {
      Node node;
      node.ParseFromString(level_itr->value);
      id_sets[level].insert(node.id());
    }
  }
}

static void GetLeaf(DistTree &dist_tree, int num, std::vector<int64_t> &ids) {
  int level = dist_tree.max_level() - 1;
  int step = static_cast<size_t>(pow(dist_tree.branch(), level)) / num;
  int ac_step = 0;
  for (auto s = dist_tree.LevelIterator(level);
       s != dist_tree.LevelEnd(level); ++s) {
    ++ac_step;
  }
  if (level < dist_tree.max_level() - 1) {
    ASSERT_TRUE(ac_step == step);
  } else {
    ASSERT_TRUE(ac_step < step);
  }
  if (step < 0) step = 0;
  auto level_itr = dist_tree.LevelIterator(level);
  auto level_end = dist_tree.LevelEnd(level);
  for (int sp = 0; sp < num; ++sp) {
    int step_num = Rand(ac_step);
    while (step_num-- > 0) {
      ASSERT_TRUE(++level_itr != level_end);
    }
    Node node;
    ASSERT_TRUE(node.ParseFromString(level_itr->value));
    ids.push_back(node.id());
  }
}

static void CreateSampleGroup(int feature_line_num,
                              int feature_num,
                              int feature_value_num,
                              DistTree &dist_tree,
                              SampleGroup *sg) {
  srand((unsigned) time(nullptr));
  FeatureTable *feature_table = sg->add_feature_tables();
  for (int i = 0; i < feature_line_num; ++i) {
    FeatureLine *feature_line = feature_table->add_feature_lines();
    for (int j = 0; j < feature_num; ++j) {
      Feature *feature = feature_line->add_features();
      feature->set_type(FeatureType::kSparse);
      if (j == 0) feature->set_name("train_unit_id");
      else feature->set_name("fake" + std::to_string(j));
      std::vector<int64_t> ids;
      GetLeaf(dist_tree, feature_value_num, ids);
      for (int k = 0; k < feature_value_num; ++k) {
        FeatureValue *feature_value = feature->add_values();
        feature_value->set_key(ids[k]);
        feature_value->set_value(1.F);
      }
    }
  }
  int label_size = feature_line_num * feature_num * feature_value_num;
  for (int i = 0; i < label_size; ++i) {
    Label *label = sg->add_labels();
    label->add_values(Rand(1));
  }
}

static void CheckEqualFeature(const Feature &ori_feature,
                              const Feature &feature) {
  ASSERT_EQ(ori_feature.values_size(), feature.values_size());
  for (int k = 0; k < feature.values_size(); ++k) {
    const FeatureValue &ori_feature_value = ori_feature.values(k);
    const FeatureValue &feature_value = feature.values(k);
    ASSERT_EQ(ori_feature_value.key(), feature_value.key());
    ASSERT_FLOAT_EQ(ori_feature_value.value(), feature_value.value());
  }
}

static void CheckEqualTable(const FeatureTable &ori_feature_table,
                            const FeatureTable &feature_table) {
  ASSERT_EQ(ori_feature_table.feature_lines_size(),
            feature_table.feature_lines_size());
  for (int i = 0; i < feature_table.feature_lines_size(); ++i) {
    const FeatureLine &ori_feature_line = ori_feature_table.feature_lines(i);
    const FeatureLine &feature_line = feature_table.feature_lines(i);
    ASSERT_EQ(ori_feature_line.features_size(), feature_line.features_size());
    for (int j = 0; j < feature_line.features_size(); ++j) {
      const Feature &ori_feature = ori_feature_line.features(j);
      const Feature &feature = feature_line.features(j);
      ASSERT_EQ(ori_feature.name(), feature.name());
      CheckEqualFeature(ori_feature, feature);
    }
  }
}

static void CheckResult(const SampleGroup *ori_sg,
                        const SampleGroup *sg,
                        DistTree &dist_tree,
                        const std::vector<std::set<int64_t>> &id_sets,
                        const int layer_counts[],
                        int layer_counts_sum) {
  printf("#### CheckResult BEGIN\n");
  ASSERT_EQ(ori_sg->feature_tables_size(), 1);
  ASSERT_EQ(sg->feature_tables_size(), 2);
  CheckEqualTable(ori_sg->feature_tables(0), sg->feature_tables(1));

  const FeatureTable &ori_feature_table = ori_sg->feature_tables(0);
  const FeatureTable &feature_table = sg->feature_tables(0);
  ASSERT_EQ(feature_table.feature_lines_size(),
            ori_feature_table.feature_lines_size() * layer_counts_sum);
  ASSERT_EQ(sg->labels_size(),
            ori_feature_table.feature_lines_size() * layer_counts_sum);
  // ori_sg ->                    x lines -> y features -> 1 values
  //     sg ->   x*layer_counts_sum lines -> 1 features -> 1 values
  //             label.shape = [ x*layer_counts_sum, 2 ]
  // layer_counts_sum: ancestors[0] neighbor...(layer_counts_sum[15])
  //                   ancestors[1] neighbor...(layer_counts_sum[14])
  //                   ...
  //                   ancestors[14] neighbor...(layer_counts_sum[1])
  // ancestors[0] = leaf
  for (int i = 0; i < ori_feature_table.feature_lines_size(); ++i) {
    const FeatureLine &ori_feature_line = ori_feature_table.feature_lines(i);
    int y = 1;
    for (int j = 0; j < y; ++j) {
      const Feature &ori_feature = ori_feature_line.features(j);
      int begin = (i * y + j) * layer_counts_sum;
      const FeatureLine
          &begin_feature_line = feature_table.feature_lines(begin);
      ASSERT_EQ(begin_feature_line.features_size(), 1);
      ASSERT_EQ(begin_feature_line.features(0).name(), "unit_id_expand");
      CheckEqualFeature(ori_feature, begin_feature_line.features(0));
      for (int k = 1; k < layer_counts_sum; ++k) {
        const FeatureLine
            &feature_line = feature_table.feature_lines(begin + k);
        ASSERT_EQ(feature_line.features_size(), 1);
        ASSERT_EQ(feature_line.features(0).name(), "unit_id_expand");
        const Feature &feature = feature_line.features(0);
        ASSERT_EQ(feature.values_size(), 1);
      }
      TreeNode tree_node = dist_tree.NodeById(ori_feature.values(0).key());
      std::vector<TreeNode> ancestors = dist_tree.Ancestors(tree_node);
      ASSERT_EQ(ancestors.size(), dist_tree.max_level() - 1);  // no root
      int idx = begin;
      for (int m = 0; m < ancestors.size(); ++m) {
        Node node;
        node.ParseFromString(ancestors[m].value);
        const Feature &feature =
            feature_table.feature_lines(idx++).features(0);  // ancestor
        ASSERT_EQ(feature.name(), "unit_id_expand");
        ASSERT_EQ(feature.values_size(), 1);
        ASSERT_EQ(feature.values(0).key(), node.id());
        const int level = ancestors.size() - m;
        const int idx_step = layer_counts[level];
        std::set<int64_t> result_id_set;
        for (int n = 0; n < idx_step; ++n) {
          const Feature &feature =
              feature_table.feature_lines(idx++).features(0);  // neighbor
          ASSERT_EQ(feature.name(), "unit_id_expand");
          ASSERT_EQ(feature.values_size(), 1);
          int64_t key = feature.values(0).key();
          ASSERT_NE(key, node.id());
          ASSERT_TRUE(id_sets[level].find(key) != id_sets[level].end());
          ASSERT_TRUE(result_id_set.find(key) == result_id_set.end());
          result_id_set.insert(key);
        }
      }
    }

    int begin = i * layer_counts_sum;
    int idx = begin;
    for (int m = 0; m < dist_tree.max_level() - 1; ++m) {
      const Label &label = sg->labels(idx++);
      ASSERT_EQ(label.values_size(), 2);
      ASSERT_FLOAT_EQ(label.values(0), 0.F);
      ASSERT_FLOAT_EQ(label.values(1), 1.F);
      const int level = dist_tree.max_level() - 1 - m;
      const int idx_step = layer_counts[level];
      for (int n = 0; n < idx_step; ++n) {
        const Label &label = sg->labels(idx++);
        ASSERT_EQ(label.values_size(), 2);
        ASSERT_FLOAT_EQ(label.values(0), 1.F);
        ASSERT_FLOAT_EQ(label.values(1), 0.F);
      }
    }
  }
  printf("#### CheckResult END\n");
}

TEST(TdmOp, TestExpand) {
  Store *store = Store::NewStore("");
  store->LoadData("../../test/test_data/movielens_tree.pb");

  DistTree &dist_tree = tdm::DistTree::GetInstance();
  dist_tree.
  set_store(store);
  ASSERT_TRUE(dist_tree.Load());
  PrintTree(dist_tree);
  for (int i = 0; i <= 10; ++i) {
    std::cout << "Test loop " << i << std::endl;
    std::vector<std::set<int64_t>> id_sets(dist_tree.max_level());
    BuildIdSets(dist_tree, id_sets);
  }

  std::vector<std::set<int64_t>> id_sets(dist_tree.max_level());
  BuildIdSets(dist_tree, id_sets);
  ASSERT_EQ(dist_tree.max_level(),16);
  const int layer_counts[dist_tree.max_level()] =
    {0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383,
     20708};
  int layer_counts_sum = 0;
  std::string layer_counts_str = "";
  for (int c = 0; c<dist_tree.max_level(); ++c) {
    layer_counts_sum += layer_counts[c];
    layer_counts_str += std::to_string(layer_counts[c]) + std::string(",");
  }
  layer_counts_sum += dist_tree.max_level() - 1;
  layer_counts_str.resize(layer_counts_str.size() - 1);
  printf("#### layer_counts_sum=%d, layer_counts_str=%s\n",
      layer_counts_sum, layer_counts_str.c_str());
  std::map<std::string, std::string> params;
  params.insert(std::make_pair("layer_counts", layer_counts_str));

  const int feature_line_num = 31, feature_num = 5, feature_value_num = 1;
  SampleGroup *sg = new SampleGroup();;
  CreateSampleGroup(feature_line_num, feature_num, feature_value_num,
      dist_tree, sg);
  SampleGroup *ori_sg = new SampleGroup();;
  ori_sg->CopyFrom(*sg);

  TDMOP tdmop;
  ASSERT_TRUE(tdmop.Init(params));
  ASSERT_TRUE(tdmop.Run(sg));

  CheckResult(ori_sg, sg, dist_tree, id_sets, layer_counts, layer_counts_sum);

  delete sg;
  delete ori_sg;
}

}  // namespace tdm

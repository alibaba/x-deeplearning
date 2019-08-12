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

#include "xdl/data_io/parser/parse_v4.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "gtest/gtest.h"


namespace xdl {
namespace io {

TEST(ParseV4Test, TestRun) {
  v4::SampleMeta meta;
  meta.set_sample_count(1);
  meta.set_data_source_name("XXX");
  meta.set_batch_type(v4::kTrainBatch);

  {
    auto comm = meta.add_data_block_meta();
    auto ncomm = meta.add_data_block_meta();

    comm->set_data_block_type(v4::kCommonFeature);
    ncomm->set_data_block_type(v4::kNCommonFeature);

    auto fgm = comm->add_feature_group_meta();
    fgm->set_feature_group_name("comm1");
    fgm->set_feature_type(v4::kKeyValue);

    fgm = ncomm->add_feature_group_meta();
    fgm->set_feature_group_name("ncomm1");
    fgm->set_feature_type(v4::kKeyValue);
  }

  std::string contents;
  EXPECT_TRUE(meta.SerializeToString(&contents));

  int N = 10;

  Schema schema;
  schema.label_count_ = 2;
  ParseV4 p(&schema);
  EXPECT_TRUE(p.InitMeta(contents));

  v4::SampleGroup v4sg;

  auto ncomm = v4sg.add_data_block();
  ncomm->set_data_block_type(v4::kNCommonFeature);

  auto label = v4sg.add_data_block();
  label->set_data_block_type(v4::kLabel);

  auto skey = v4sg.add_data_block();
  skey->set_data_block_type(v4::kSampleInfo);

  for (int i = 0; i < N; ++i) {
    auto fb = ncomm->add_feature_block();
    auto fg = fb->add_feature_group();
    fg->set_feature_index(0);
    auto kv = fg->add_kv_feature();
    kv->set_id(i);
    kv->set_value(0.2*i);

    auto lb = label->add_label_block();
    lb->add_data(0.1*i);
    lb->add_data(0.1*i);

    auto sb = skey->add_sample_info_block();
    sb->set_info(std::to_string(i));
  }

  auto comm= v4sg.add_data_block();
  comm->set_data_block_type(v4::kCommonFeature);
  auto fb = comm->add_feature_block();
  auto fg = fb->add_feature_group();
  fg->set_feature_index(0);
  auto kv = fg->add_kv_feature();
  kv->set_id(1);
  kv->set_value(0.1);

  std::cout << v4sg.ShortDebugString() << std::endl;
  EXPECT_TRUE(v4sg.SerializeToString(&contents));

  int t = contents.size();
  std::string str;
  str.assign((char *)&t, 4);
  str += contents;

  size_t n = p.GetSize(str.c_str(), str.size());
  EXPECT_EQ(contents.size()+4, n);

  auto sgroup = p.Run(str.c_str(), str.size());
  EXPECT_NE(nullptr, sgroup);

  auto sg = sgroup->Get();
  std::cout << sg->ShortDebugString() << std::endl;

  EXPECT_EQ(N, sg->labels_size());
  EXPECT_EQ(N, sg->sample_ids_size());
  EXPECT_EQ(2, sg->feature_tables_size());

  auto ft0 = sg->feature_tables(0);
  EXPECT_EQ(N, ft0.feature_lines_size());

  auto ft1 = sg->feature_tables(1);
  EXPECT_EQ(1, ft1.feature_lines_size());

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(2, sg->labels(i).values_size());
    EXPECT_FLOAT_EQ(0.1*i, sg->labels(i).values(0));
    EXPECT_FLOAT_EQ(0.1*i, sg->labels(i).values(1));

    EXPECT_STREQ((std::to_string(i)).c_str(), sg->sample_ids(i).c_str());

    /* table0 */
    auto fl = ft0.feature_lines(i);
    EXPECT_EQ(0, fl.refer());
    EXPECT_EQ(1, fl.features_size());

    auto f = fl.features(0);
    EXPECT_EQ(kSparse, f.type());
    EXPECT_STREQ("ncomm1", f.name().c_str());
    EXPECT_EQ(1, f.values_size());

    auto fv = f.values(0);
    EXPECT_EQ(i, fv.key());
    EXPECT_FLOAT_EQ(0.2*i, fv.value());
    ++i;
  }

  /* table1 */
  auto fl = ft1.feature_lines(0);
  EXPECT_EQ(1, fl.features_size());

  auto f = fl.features(0);
  EXPECT_EQ(kSparse, f.type());
  EXPECT_STREQ("comm1", f.name().c_str());
  EXPECT_EQ(1, f.values_size());

  auto fv = f.values(0);
  EXPECT_EQ(1, fv.key());
  EXPECT_FLOAT_EQ(0.1, fv.value());
}

}  // namespace io
}  // namespace xdl

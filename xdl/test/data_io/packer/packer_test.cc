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

#include "xdl/data_io/packer/packer.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "xdl/data_io/constant.h"
#include "xdl/data_io/pool.h"
#include "xdl/core/framework/cpu_device.h"

namespace xdl {
namespace io {

class PackerTest: public ::testing::Test {
 public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  static SGroup *InitSGroup(size_t size);

  static const size_t kBatchSize;
  static const size_t kLabelCount;
  static const size_t kTableCount;
  static const size_t kFeatureCount;

  static Schema schema_;
  static Packer *packer_;
};

const size_t PackerTest::kBatchSize = 4;
const size_t PackerTest::kLabelCount = 6;
const size_t PackerTest::kTableCount = 3;
const size_t PackerTest::kFeatureCount = 16;

Schema PackerTest::schema_;
Packer *PackerTest::packer_ = nullptr;

void PackerTest::SetUpTestCase() {
  schema_.batch_size_ = kBatchSize;
  schema_.label_count_ = kLabelCount;
  for (int ktable = 0; ktable < kTableCount; ++ktable) {
    for (int c = 0; c < kFeatureCount; ++c) {
      FeatureOption *s = new FeatureOption();
      s->set_name(std::to_string(ktable)+"u"+std::to_string(c));
      s->set_type(kSparse);
      s->set_table(ktable);
      schema_.Add(s);

      FeatureOption *d = new FeatureOption();
      d->set_name(std::to_string(ktable)+"a"+std::to_string(c));
      d->set_type(kDense);
      d->set_nvec(c+1);
      d->set_table(ktable);
      schema_.Add(d);
    }
  }

  packer_ = new Packer(&schema_, new CpuDevice());

  EXPECT_TRUE(packer_->Init());
}

SGroup *PackerTest::InitSGroup(size_t size) {
  auto sgroup = SGroupPool::Get()->Acquire();
  auto sg = sgroup->New();
  /// init skey & label
  for (int i = 0; i < size; ++i) {
    sg->add_sample_ids("sk"+std::to_string(i));
    auto label = sg->add_labels();
    for (int j = 0; j < kLabelCount; ++j) {
      label->add_values(j);
    }
  }
  /// init feature
  for (int ktable = 0; ktable < kTableCount; ++ktable) {
    auto ft = sg->add_feature_tables();
    for (int i = 0; i < size; ++i) {
      auto fl = ft->add_feature_lines();
      /// refer
      if (ktable < kTableCount - 1) {
        fl->set_refer((i+1)/2);
      }
    }
    size = size / 2 + 1;
  }
  sgroup->Reset();
  return sgroup;
}

void PackerTest::TearDownTestCase() {
  delete packer_;
}

TEST_F(PackerTest, TestEmpty) {
  auto batchs = packer_->Run((SGroup *)END);
  EXPECT_EQ(0, batchs.size());
}

TEST_F(PackerTest, TestLess) {
  auto sgroup = InitSGroup(1);
  auto batchs = packer_->Run(sgroup);
  EXPECT_EQ(0, batchs.size());

  sgroup = InitSGroup(1);
  batchs = packer_->Run(sgroup);
  EXPECT_EQ(0, batchs.size());

  batchs = packer_->Run((SGroup *)END);
  EXPECT_EQ(1, batchs.size());
}

TEST_F(PackerTest, TestLess2Equal) {
  auto sgroup = InitSGroup(1);
  auto batchs = packer_->Run(sgroup);
  EXPECT_EQ(0, batchs.size());

  sgroup = InitSGroup(kBatchSize-1);
  batchs = packer_->Run(sgroup);
  EXPECT_EQ(1, batchs.size());

  batchs = packer_->Run((SGroup *)END);
  EXPECT_EQ(0, batchs.size());
}

TEST_F(PackerTest, TestLess2More) {
  auto sgroup = InitSGroup(1);
  auto batchs = packer_->Run(sgroup);
  EXPECT_EQ(0, batchs.size());

  sgroup = InitSGroup(kBatchSize);
  batchs = packer_->Run(sgroup);
  EXPECT_EQ(1, batchs.size());

  batchs = packer_->Run((SGroup *)END);
  EXPECT_EQ(1, batchs.size());
}

TEST_F(PackerTest, TestMore) {
  auto sgroup = InitSGroup(kBatchSize*5);
  auto batchs = packer_->Run(sgroup);
  EXPECT_EQ(5, batchs.size());

  batchs = packer_->Run((SGroup *)END);
  EXPECT_EQ(0, batchs.size());
}

TEST_F(PackerTest, TestTruncate) {
  schema_.split_group_ = false;
  auto sgroup = InitSGroup(kBatchSize*5);
  auto batchs = packer_->Run(sgroup);
  EXPECT_EQ(1, batchs.size());

  batchs = packer_->Run((SGroup *)END);
  EXPECT_EQ(0, batchs.size());
}

}  // io
}  // xdl


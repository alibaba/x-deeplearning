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

#include "xdl/data_io/packer/pack_label.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "xdl/data_io/pool.h"
#include "xdl/core/framework/cpu_device.h"


namespace xdl {
namespace io {

class PackLabelTest: public ::testing::Test {
 public:
  static void SetUpTestCase() {
    dev_ = new CpuDevice();
    schema_.batch_size_ = kBatchSize;
    schema_.label_count_ = kLabelCount;
    pack_ = new PackLabel(dev_, &schema_);
    batch_ = BatchPool::Get()->Acquire();
    EXPECT_NE(nullptr, batch_);
    std::srand(std::time(nullptr));

    EXPECT_TRUE(pack_->Init(batch_));

    int c = std::rand() % 32 + 1;
    for (int i = 0; i < c; ++i) {
      Label *lb = sg1_.add_labels();
      std::vector<float> l;
      for (int j = 0; j < kLabelCount; ++j) {
        l.push_back(i+j*0.1);
        lb->add_values(i+j*0.1);
      }
      s1_.push_back(l);
    }

    c = std::rand() % 32 + 1;
    for (int i = 0; i < c; ++i) {
      Label *lb = sg2_.add_labels();
      std::vector<float> l;
      for (int j = 0; j < kLabelCount; ++j) {
        l.push_back(i*1000+j);
        lb->add_values(i*1000+j);
      }
      s2_.push_back(l);
    }
  }

  static void TearDownTestCase() {
    BatchPool::Get()->Release(batch_);
    batch_ = nullptr;
    delete pack_;
    pack_ = nullptr;
  }

  static void TestStat();
  static void TestSetup();
  static void TestRun();

  static const size_t kBatchSize;
  static const size_t kLabelCount;

  static Device *dev_;
  static Schema schema_;

  static Pack *pack_;
  static Batch *batch_;

  static std::vector<std::vector<float>> s1_;
  static std::vector<std::vector<float>> s2_;
  static SampleGroup sg1_;
  static SampleGroup sg2_;
};
const size_t PackLabelTest::kBatchSize = 64;
const size_t PackLabelTest::kLabelCount = 6;

Device *PackLabelTest::dev_ = nullptr;
Schema PackLabelTest::schema_;

Pack *PackLabelTest::pack_ = nullptr;
Batch *PackLabelTest::batch_ = nullptr;

std::vector<std::vector<float>> PackLabelTest::s1_;
std::vector<std::vector<float>> PackLabelTest::s2_;
SampleGroup PackLabelTest::sg1_;
SampleGroup PackLabelTest::sg2_;

void PackLabelTest::TestStat() {
  PParam pparam;

  pparam.begin_ = 0;
  pparam.end_ = s1_.size();
  pparam.labels_ = &sg1_.labels();
  pparam.isgroup_ = 0;
  pack_->Stat(pparam);

  pparam.begin_ = 0;
  pparam.end_ = s2_.size();
  pparam.labels_ = &sg2_.labels();
  pparam.isgroup_ = 1;
  pack_->Stat(pparam);
}

void PackLabelTest::TestSetup() {
  EXPECT_TRUE(pack_->Setup());

  auto blk = batch_->GetMutable(kLabelName);
  EXPECT_NE(nullptr, blk);
  EXPECT_NE(nullptr, blk->ts_[Block::kValue]);

  auto dims = blk->ts_[Block::kValue]->Shape().Dims();
  EXPECT_EQ(2, dims.size());
  EXPECT_EQ(kBatchSize, dims[0]);
  EXPECT_EQ(kLabelCount, dims[1]);

  EXPECT_EQ(1, blk->ts_count_);
  EXPECT_EQ(1, batch_->ts_count_);
}

void PackLabelTest::TestRun() {
  PParam pparam;

  pparam.begin_ = 0;
  pparam.end_ = s1_.size();
  pparam.labels_ = &sg1_.labels();
  pparam.isgroup_ = 0;
  pack_->Run(pparam);

  pparam.begin_ = 0;
  pparam.end_ = s2_.size();
  pparam.labels_ = &sg2_.labels();
  pparam.isgroup_ = 1;
  pack_->Run(pparam);

  auto blk = batch_->GetMutable(kLabelName);
  auto labels = blk->ts_[Block::kValue]->Raw<float>();
  auto dims = blk->ts_[Block::kValue]->Shape().Dims();

  EXPECT_EQ(1, blk->ts_count_);
  EXPECT_EQ(1, batch_->ts_count_);

  for (int i = 0; i < dims[0]; ++i) {
    if (i < s1_.size() + s2_.size()) {
      std::vector<float> &l = i < s1_.size() ? s1_[i] : s2_[i-s1_.size()];
      for (int j = 0; j < dims[1]; ++j) {
        EXPECT_EQ(l[j], labels[i*kLabelCount + j]);
      }
    } else {
    }
  }
}

TEST_F(PackLabelTest, Stat) {
  TestStat();
}

TEST_F(PackLabelTest, Setup) {
  TestSetup();
}

TEST_F(PackLabelTest, Run) {
  TestRun();
}

}  // io
}  // xdl


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

#include "xdl/data_io/packer/pack_skey.h"

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

class PackSKeyTest: public ::testing::Test {
 public:
  static void SetUpTestCase() {
    dev_ = new CpuDevice();
    schema_.batch_size_ = kBatchSize;
    pack_ = new PackSKey(dev_, &schema_);
    batch_ = BatchPool::Get()->Acquire();
    EXPECT_NE(nullptr, batch_);
    std::srand(std::time(nullptr));

    EXPECT_TRUE(pack_->Init(batch_));

    int c = std::rand() % 32 + 1;
    for (int i = 0; i < c; ++i) {
      auto s = std::to_string(i);
      skey_len_max_ = std::max(skey_len_max_, s.size()+1);
      s1_.push_back(s);
      sg1_.add_sample_ids(s);
    }

    c = std::rand() % 32 + 1;
    for (int i = 0; i < c; ++i) {
      auto s = std::to_string(i*1000);
      skey_len_max_ = std::max(skey_len_max_, s.size()+1);
      s2_.push_back(s);
      sg2_.add_sample_ids(s);
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

  static Device *dev_;
  static Schema schema_;

  static Pack *pack_;
  static Batch *batch_;
  static size_t skey_len_max_;

  static std::vector<std::string> s1_;
  static std::vector<std::string> s2_;
  static SampleGroup sg1_;
  static SampleGroup sg2_;
};
const size_t PackSKeyTest::kBatchSize = 64;

Device *PackSKeyTest::dev_ = nullptr;
Schema PackSKeyTest::schema_;

Pack *PackSKeyTest::pack_ = nullptr;
Batch *PackSKeyTest::batch_ = nullptr;
size_t PackSKeyTest::skey_len_max_ = 0;

std::vector<std::string> PackSKeyTest::s1_;
std::vector<std::string> PackSKeyTest::s2_;
SampleGroup PackSKeyTest::sg1_;
SampleGroup PackSKeyTest::sg2_;

void PackSKeyTest::TestStat() {
  PParam pparam;

  pparam.begin_ = 0;
  pparam.end_ = s1_.size();
  pparam.sample_ids_ = &sg1_.sample_ids();
  pparam.isgroup_ = 0;
  pack_->Stat(pparam);

  pparam.begin_ = 0;
  pparam.end_ = s2_.size();
  pparam.sample_ids_ = &sg2_.sample_ids();
  pparam.isgroup_ = 1;
  pack_->Stat(pparam);
}

void PackSKeyTest::TestSetup() {
  EXPECT_TRUE(pack_->Setup());

  auto blk = batch_->GetMutable(kSKeyName);
  EXPECT_NE(nullptr, blk);
  EXPECT_NE(nullptr, blk->ts_[Block::kSBuf]);

  auto dims = blk->ts_[Block::kSBuf]->Shape().Dims();
  EXPECT_EQ(2, dims.size());
  EXPECT_EQ(kBatchSize, dims[0]);
  EXPECT_EQ(skey_len_max_, dims[1]);

  EXPECT_EQ(1, blk->ts_count_);
  EXPECT_EQ(1, batch_->ts_count_);
}

void PackSKeyTest::TestRun() {
  PParam pparam;

  pparam.begin_ = 0;
  pparam.end_ = s1_.size();
  pparam.sample_ids_ = &sg1_.sample_ids();
  pparam.isgroup_ = 0;
  pack_->Run(pparam);

  pparam.begin_ = 0;
  pparam.end_ = s2_.size();
  pparam.sample_ids_ = &sg2_.sample_ids();
  pparam.isgroup_ = 1;
  pack_->Run(pparam);

  auto blk = batch_->GetMutable(kSKeyName);
  auto sbuf = (char *)blk->ts_[Block::kSBuf]->Raw<int8_t>();
  auto dims = blk->ts_[Block::kSBuf]->Shape().Dims();
  EXPECT_EQ(2, dims.size());
  EXPECT_EQ(kBatchSize, dims[0]);
  EXPECT_EQ(skey_len_max_, dims[1]);

  EXPECT_EQ(1, blk->ts_count_);
  EXPECT_EQ(1, batch_->ts_count_);

  for (int n = 0; n < dims[0]; ++n) {
    if (n < s1_.size() + s2_.size()) {
      std::string &s = n < s1_.size() ? s1_[n] : s2_[n-s1_.size()];
      EXPECT_EQ(s.size(), strlen(&sbuf[n*dims[1]]));
      EXPECT_STREQ(s.c_str(), &sbuf[n*dims[1]]);
    }
  }
}

TEST_F(PackSKeyTest, Stat) {
  TestStat();
}

TEST_F(PackSKeyTest, Setup) {
  TestSetup();
}

TEST_F(PackSKeyTest, Run) {
  TestRun();
}

}  // io
}  // xdl


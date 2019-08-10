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

#include "xdl/data_io/packer/pack_feature.h"

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


class PackFeatureTest: public ::testing::Test {
  static const size_t kBatchSize;
  static const size_t kSGCount;
  static const size_t kTableCount;

 public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  static size_t sample_count(size_t ktable);
  static size_t batch_size(size_t ktable);

  static void TestStat();
  static void TestSetup();
  static void TestRun();

  static PackFeature *pack_;
  static Batch *batch_;


 private:
  static void CheckIndicator();
  static void CheckFeature();

  static Device *dev_;
  static Schema schema_;

  static std::vector<SampleGroup> sgs_;
};

const size_t PackFeatureTest::kBatchSize = 8192;
//const size_t PackFeatureTest::kBatchSize = 4;
const size_t PackFeatureTest::kSGCount = 128;
//const size_t PackFeatureTest::kSGCount = 2;
const size_t PackFeatureTest::kTableCount = 2;

Device *PackFeatureTest::dev_ = nullptr;
Schema PackFeatureTest::schema_;

PackFeature *PackFeatureTest::pack_ = nullptr;
Batch *PackFeatureTest::batch_ = nullptr;

std::vector<SampleGroup> PackFeatureTest::sgs_;

size_t PackFeatureTest::sample_count(size_t ktable) {
  size_t c = kBatchSize/kSGCount - 1;
  for (int k = 0; k < ktable+1; ++k) {
    c = c / 2 + 1;
  }
  return c;
}

size_t PackFeatureTest::batch_size(size_t ktable) {
  size_t c = sample_count(ktable);
  return c * kSGCount;
}

void PackFeatureTest::SetUpTestCase() {
  dev_ = new CpuDevice();
  schema_.batch_size_ = kBatchSize;
  schema_.padding_ = false;

  for (int ktable = 0; ktable < kTableCount; ++ktable) {
    FeatureOption *s = new FeatureOption();
    s->set_name(std::to_string(ktable)+"s");
    s->set_type(kSparse);
    s->set_table(ktable);
    schema_.Add(s);

    FeatureOption *d = new FeatureOption();
    d->set_name(std::to_string(ktable)+"a");
    d->set_type(kDense);
    d->set_nvec(2);
    d->set_table(ktable);
    schema_.Add(d);
  }

  sgs_.resize(kSGCount);

  for (int i = 0; i < kSGCount; ++i) {
    auto &sg = sgs_[i];
    int count = kBatchSize/kSGCount - 1;
    for (int ktable = 0; ktable < kTableCount; ++ktable) {
      count = count / 2 + 1;
      auto ft = sg.add_feature_tables();
      for (int n = 0; n < count; ++n) {
        auto fl = ft->add_feature_lines();
        auto f = fl->add_features();
        f->set_name(std::to_string(ktable)+"d");
        f->set_type(kDense);
        auto v = f->add_values();
        for (int m = 0; m < 2; ++m) {
          v->add_vector(0.1*m);
        }
        f = fl->add_features();
        f->set_name(std::to_string(ktable)+"s");
        f->set_type(kSparse);
        v = f->add_values();
        v->set_key(1);
        v->set_value(0.6);
        /// refer
        if (ktable < kTableCount - 1) {
          fl->set_refer((n+1)/2);
        }
      }
    }
  }

  pack_ = new PackFeature(dev_, &schema_);
}

void PackFeatureTest::TearDownTestCase() {
  BatchPool::Get()->Release(batch_);
  batch_ = nullptr;
  delete pack_;
  pack_ = nullptr;
}

void PackFeatureTest::TestStat() {
  PParam pparam;

  for (int i = 0; i < kSGCount; ++i) {
    int count = kBatchSize/kSGCount - 1;
    pparam.begin_ = 0;
    pparam.end_ = sgs_[i].feature_tables(0).feature_lines_size();
    for (int ktable = 0; ktable < kTableCount; ++ktable) {
      count = count / 2 + 1;
      pparam.ftable_ = &sgs_[i].feature_tables(ktable);
      pparam.ktable_ = ktable;
      pparam.isgroup_ = i;
      EXPECT_GE(pparam.begin_, 0);
      EXPECT_LE(pparam.end_, count);
      //std::cout << "stat[" << pparam.isgroup_ << ", " << ktable <<  "] (0)" << pparam.begin_ 
      //    << " -> " <<  pparam.end_ << "(" << pparam.ftable_->feature_lines_size() << ")" << std::endl;
      auto range = pack_->Stat(pparam);
      pparam.begin_ = range.first;
      pparam.end_ = range.second;
    }
  }

}

void PackFeatureTest::TestSetup() {
  ASSERT_TRUE(pack_->Setup());

  for (auto &it: schema_.feature_opts()) {
    auto opt = it.second;

    auto blk = batch_->GetMutable(opt->name());
    ASSERT_NE(nullptr, blk);
    ASSERT_NE(nullptr, blk->ts_[Block::kValue]);
    auto vdims = blk->ts_[Block::kValue]->Shape().Dims();

    auto ktable = opt->table();
    size_t bs = batch_size(ktable);

    if (opt->type() == kSparse) {
      ASSERT_NE(nullptr, blk->ts_[Block::kKey]);
      auto kdims = blk->ts_[Block::kKey]->Shape().Dims();
      ASSERT_NE(nullptr, blk->ts_[Block::kSegment]);
      auto sdims = blk->ts_[Block::kSegment]->Shape().Dims();
      ASSERT_EQ(bs, sdims[0]);
      ASSERT_EQ(3, blk->ts_count_);
    } else {
      ASSERT_EQ(2, vdims.size());
      ASSERT_EQ(bs, vdims[0]);
      ASSERT_EQ(1, blk->ts_count_);
    }
  }

  ASSERT_EQ(kTableCount*(3+1) + kTableCount-1, batch_->ts_count_);
}


void PackFeatureTest::TestRun() {
  PParam pparam;

  for (int i = 0; i < kSGCount; ++i) {
    pparam.begin_ = 0;
    pparam.end_ = sgs_[i].feature_tables(0).feature_lines_size();
    for (int ktable = 0; ktable < kTableCount; ++ktable) {
      pparam.ftable_ = &sgs_[i].feature_tables(ktable);
      pparam.ktable_ = ktable;
      pparam.isgroup_ = i;
      //std::cout << "run[" << pparam.isgroup_ << ", " << ktable <<  "] (0)" << pparam.begin_ 
      //    << " -> " <<  pparam.end_ << "(" << pparam.ftable_->feature_lines_size() << ")" << std::endl;
      auto range = pack_->Run(pparam);
      pparam.begin_ = range.first;
      pparam.end_ = range.second;
    }
  }
}

TEST_F(PackFeatureTest, Run) {

    batch_ = BatchPool::Get()->Acquire();
    EXPECT_NE(nullptr, batch_);

    EXPECT_TRUE(pack_->Init(batch_));

    TestStat();
    TestSetup();
    TestRun();

    batch_->Reuse();
    batch_ = nullptr;

    //std::cout << "cycles: " << pack_->cycles_ << std::endl;
}

}  // io
}  // xdl

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

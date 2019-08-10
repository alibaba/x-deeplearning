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


class PackCutoffTest: public ::testing::Test {
  static const size_t kBatchSize;

 public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  static void TestStat();
  static void TestSetup();
  static void TestRun();

  static PackFeature *pack_;
  static Batch *batch_;


 private:
  static void CheckFeature();

  static Device *dev_;
  static Schema schema_;
  static SampleGroup sg_;
};

const size_t PackCutoffTest::kBatchSize = 4;

Device *PackCutoffTest::dev_ = nullptr;
Schema PackCutoffTest::schema_;

PackFeature *PackCutoffTest::pack_ = nullptr;
Batch *PackCutoffTest::batch_ = nullptr;

SampleGroup PackCutoffTest::sg_;

void PackCutoffTest::SetUpTestCase() {
  dev_ = new CpuDevice();
  schema_.batch_size_ = kBatchSize;

  for (auto fn: {"s", "l", "sr", "lr"}) {
    FeatureOption *f = new FeatureOption();
    f->set_name(fn);
    f->set_type(kSparse);
    f->set_table(0);
    f->set_serialized(true);
    int cutoff;
    if (fn[0] == 's') {
      cutoff = 3;
    } else {
      ASSERT_TRUE(fn[0] == 'l');
      cutoff = 5;
    }
    if (strlen(fn) == 2) {
      ASSERT_TRUE(fn[1] == 'r');
      cutoff = -cutoff;
    }
    f->set_cutoff(cutoff);
    schema_.Add(f);
  }

  auto ft = sg_.add_feature_tables();
  for (int i = 0; i < kBatchSize; ++i) {
    auto fl = ft->add_feature_lines();
    for (auto fn: {"s", "l", "sr", "lr"}) {
      auto f = fl->add_features();
      f->set_name(fn);
      f->set_type(kSparse);
      for (int m = 1; m < 5; ++m) {
        auto v = f->add_values();
        v->set_key(m);
        v->set_value(m*0.1);
      }
    }
  }

  pack_ = new PackFeature(dev_, &schema_);
}

void PackCutoffTest::TearDownTestCase() {
  BatchPool::Get()->Release(batch_);
  batch_ = nullptr;
  delete pack_;
  pack_ = nullptr;
}


void PackCutoffTest::CheckFeature() {
  for (auto fn: {"s", "l", "sr", "lr"}) {
    auto blk = batch_->Get(fn);
    ASSERT_NE(nullptr, blk);

    auto key = blk->ts_[Block::kKey];
    auto value = blk->ts_[Block::kValue];
    auto seg = blk->ts_[Block::kSegment];

    uint64_t *keys = key->Raw<uint64_t>();
    float *vals = value->Raw<float>();
    uint32_t *segs = seg->Raw<uint32_t>();

    int cutoff;
    if (fn[0] == 's') {
      cutoff = 3;
    } else {
      ASSERT_TRUE(fn[0] == 'l');
      cutoff = 5;
    }
    if (strlen(fn) == 2) {
      ASSERT_TRUE(fn[1] == 'r');
      cutoff = -cutoff;
    }

    //std::cout << fn << " cutoff " << cutoff << std::endl;
    for (int i = 0, n = 0, c = segs[n]; i < key->Shape()[0]; ++i, --c) {
      if (c == 0) {
        //std::cout << std::endl;
        if (++ n == kBatchSize) { break; }
        c = segs[n] - segs[n-1];
        ASSERT_EQ(std::min(abs(cutoff), 4), c);
        //std::cout << std::endl << "[" << n << "]";
      }
      //std::cout<< keys[i] << ":"  << vals[i] << " ";
      if (cutoff > 0) {
        EXPECT_EQ(i - segs[n-1] + 1, keys[i]);
        EXPECT_FLOAT_EQ((i - segs[n-1] + 1)*0.1, vals[i]);
      } else {
        //EXPECT_EQ(4 - (i - segs[n-1]), keys[i]);
        //EXPECT_FLOAT_EQ((4 - (i - segs[n-1]))*0.1, vals[i]);
        EXPECT_EQ((i - segs[n-1]) + std::max(0, 4+cutoff) + 1, keys[i]);
        EXPECT_FLOAT_EQ(((i - segs[n-1]) + std::max(0, 4+cutoff) + 1)*0.1, vals[i]);
      }
    }
    //std::cout << std::endl;
  }
}

void PackCutoffTest::TestStat() {
  PParam pparam;

  pparam.begin_ = 0;
  pparam.end_ = sg_.feature_tables(0).feature_lines_size();
  pparam.ftable_ = &sg_.feature_tables(0);
  pparam.ktable_ = 0;
  pparam.isgroup_ = 0;
  EXPECT_GE(pparam.begin_, 0);
  EXPECT_GE(pparam.end_, kBatchSize);
  //std::cout << "stat[" << pparam.isgroup_ << ", " << ktable <<  "] (0)" << pparam.begin_ 
  //    << " -> " <<  pparam.end_ << "(" << pparam.ftable_->feature_lines_size() << ")" << std::endl;
  auto range = pack_->Stat(pparam);
  pparam.begin_ = range.first;
  pparam.end_ = range.second;
}

void PackCutoffTest::TestSetup() {
  ASSERT_TRUE(pack_->Setup());

  for (auto &it: schema_.feature_opts()) {
    auto opt = it.second;
    auto ktable = opt->table();

    auto blk = batch_->GetMutable(opt->name());
    ASSERT_NE(nullptr, blk);
    ASSERT_NE(nullptr, blk->ts_[Block::kValue]);
    ASSERT_NE(nullptr, blk->ts_[Block::kKey]);
    ASSERT_NE(nullptr, blk->ts_[Block::kSegment]);
    auto kdims = blk->ts_[Block::kKey]->Shape().Dims();
    auto sdims = blk->ts_[Block::kSegment]->Shape().Dims();
    auto fn = opt->name().c_str();
    int cutoff;
    if (fn[0] == 's') {
      cutoff = 3;
    } else {
      ASSERT_TRUE(fn[0] == 'l');
      cutoff = 5;
    }
    if (strlen(fn) == 2) {
      ASSERT_TRUE(fn[1] == 'r');
      cutoff = -cutoff;
    }
    ASSERT_EQ(cutoff, opt->cutoff());
    ASSERT_EQ(kSparse, opt->type());
    ASSERT_EQ(std::vector<size_t>({kBatchSize*std::min(4, abs(cutoff))}), kdims);
    ASSERT_EQ(std::vector<size_t>({kBatchSize}), sdims);
    ASSERT_EQ(3, blk->ts_count_);
  }

  ASSERT_EQ(12, batch_->ts_count_);
}

void PackCutoffTest::TestRun() {
  PParam pparam;

  pparam.begin_ = 0;
  pparam.end_ = sg_.feature_tables(0).feature_lines_size();
  pparam.ftable_ = &sg_.feature_tables(0);
  pparam.ktable_ = 0;
  pparam.isgroup_ = 0;
  //std::cout << "run[" << pparam.isgroup_ << ", " << ktable <<  "] (0)" << pparam.begin_ 
  //    << " -> " <<  pparam.end_ << "(" << pparam.ftable_->feature_lines_size() << ")" << std::endl;
  auto range = pack_->Run(pparam);
  pparam.begin_ = range.first;
  pparam.end_ = range.second;

  CheckFeature();

  EXPECT_EQ(12, batch_->ts_count_);
}

TEST_F(PackCutoffTest, Run) {
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

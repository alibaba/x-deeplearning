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


class PackMaskTest: public ::testing::Test {
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

const size_t PackMaskTest::kBatchSize = 4;

Device *PackMaskTest::dev_ = nullptr;
Schema PackMaskTest::schema_;

PackFeature *PackMaskTest::pack_ = nullptr;
Batch *PackMaskTest::batch_ = nullptr;

SampleGroup PackMaskTest::sg_;

void PackMaskTest::SetUpTestCase() {
  dev_ = new CpuDevice();
  schema_.batch_size_ = kBatchSize;

  FeatureOption *d = new FeatureOption();
  d->set_name("d");
  d->set_type(kDense);
  d->set_nvec(4);
  d->set_table(0);
  d->set_mask("1110");
  schema_.Add(d);

  auto ft = sg_.add_feature_tables();
  for (int i = 0; i < kBatchSize; ++i) {
    auto fl = ft->add_feature_lines();
    auto f = fl->add_features();
    f->set_name("d");
    f->set_type(kDense);
    auto v = f->add_values();
    for (int m = 0; m < 4; ++m) {
        v->add_vector(0.1*m);
    }
  }

  pack_ = new PackFeature(dev_, &schema_);
}

void PackMaskTest::TearDownTestCase() {
  BatchPool::Get()->Release(batch_);
  batch_ = nullptr;
  delete pack_;
  pack_ = nullptr;
}


void PackMaskTest::CheckFeature() {
  for (auto &kv : schema_.feature_opts()) {
    auto &opt = kv.second;
    auto blk = batch_->Get(opt->name());
    ASSERT_NE(nullptr, blk);

    auto value = blk->ts_[Block::kValue];
    ASSERT_EQ(kBatchSize, value->Shape()[0]);
    ASSERT_EQ(3, value->Shape()[1]);

    float *values = value->Raw<float>();
    for (int i = 0; i < value->Shape()[0]; ++i) {
      for (int m = 0; m < value->Shape()[1]; ++m) {
        std::cout << values[i*value->Shape()[1]+m] << ", ";
      }
      std::cout << std::endl;
    }
  }
}

void PackMaskTest::TestStat() {
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

void PackMaskTest::TestSetup() {
  ASSERT_TRUE(pack_->Setup());

  for (auto &it: schema_.feature_opts()) {
    auto opt = it.second;

    auto blk = batch_->GetMutable(opt->name());
    ASSERT_NE(nullptr, blk);
    ASSERT_NE(nullptr, blk->ts_[Block::kValue]);
    auto vdims = blk->ts_[Block::kValue]->Shape().Dims();

    auto ktable = opt->table();
    size_t bs = kBatchSize;

    ASSERT_EQ(kDense, opt->type());
    ASSERT_EQ(2, vdims.size());
    ASSERT_EQ(bs, vdims[0]);
    ASSERT_EQ(1, blk->ts_count_);
  }

  ASSERT_EQ(1, batch_->ts_count_);
}

void PackMaskTest::TestRun() {
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

  EXPECT_EQ(1, batch_->ts_count_);
}

TEST_F(PackMaskTest, Run) {
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

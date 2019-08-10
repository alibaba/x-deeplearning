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

#include "xdl/data_io/merger/merger.h"

#include <cstdlib>
#include <cmath>
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

class MergerTest: public ::testing::Test {
 public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  static SGroup *InitSGroup(size_t size);

  static const size_t kFeatureCount;

  static Schema schema_;
  static Merger *merger_;
  static Batch batch_;

  static std::vector<int64_t> i64_;
  static std::vector<std::pair<int64_t, int64_t>> i128_;
};

const size_t MergerTest::kFeatureCount = 16;

Schema MergerTest::schema_;
Merger *MergerTest::merger_ = nullptr;
Batch MergerTest::batch_;

std::vector<int64_t> MergerTest::i64_;
std::vector<std::pair<int64_t, int64_t>> MergerTest::i128_;

void MergerTest::SetUpTestCase() {
  Device *dev = new CpuDevice();
  merger_ = new Merger(&schema_, dev);
  auto a = batch_.GetMutable("a");
  a->ts_[Block::kKey] = new Tensor(dev, TensorShape({kFeatureCount, 1}), types::kInt64);
  a->ts_[Block::kSegment] = new Tensor(dev, TensorShape({kFeatureCount, 1}), types::kInt32);  
  auto keys = a->ts_[Block::kKey]->Raw<int64_t>();
  int r = kFeatureCount / 2 + 1;
  for (size_t i = 0; i < kFeatureCount; ++i) {
    int64_t k = std::rand() % r;
    keys[i] = k;
    i64_.push_back(k);
  }

  auto b = batch_.GetMutable("b");
  b->ts_[Block::kKey] = new Tensor(dev, TensorShape({kFeatureCount, 2}), types::kInt64);
  b->ts_[Block::kIndex] = new Tensor(dev, TensorShape({kFeatureCount}), types::kInt32);
  b->ts_[Block::kSegment] = new Tensor(dev, TensorShape({kFeatureCount, 1}), types::kInt32);  
  keys = b->ts_[Block::kKey]->Raw<int64_t>();
  r = std::sqrt(kFeatureCount) + 1;
  for (size_t i = 0; i < kFeatureCount; ++i) {
    int64_t k = std::rand() % r;
    int64_t k1 = std::rand() % r;
    keys[i*2] = k;
    keys[i*2+1] = k1;
    i128_.push_back(std::make_pair(k, k1));
  }

  auto c = batch_.GetMutable("c");
  c->ts_[Block::kIndex] = new Tensor(dev, TensorShape({kFeatureCount}), types::kInt32);
}

void MergerTest::TearDownTestCase() {
  delete merger_;
}

TEST_F(MergerTest, TestRun) {
  auto batch = merger_->Run(&batch_);
  auto a = batch_.Get("a");
  EXPECT_NE(nullptr, a->ts_[Block::kKey]);
  EXPECT_NE(nullptr, a->ts_[Block::kUKey]);
  EXPECT_NE(nullptr, a->ts_[Block::kIndex]);
  EXPECT_NE(nullptr, a->ts_[Block::kSIndex]);
  EXPECT_NE(nullptr, a->ts_[Block::kSSegment]);    

  auto b = batch_.Get("b");
  EXPECT_NE(nullptr, b->ts_[Block::kKey]);
  EXPECT_NE(nullptr, b->ts_[Block::kUKey]);
  EXPECT_NE(nullptr, b->ts_[Block::kIndex]);
  EXPECT_NE(nullptr, b->ts_[Block::kSIndex]);
  EXPECT_NE(nullptr, b->ts_[Block::kSSegment]);

  auto c = batch_.Get("c");
  EXPECT_EQ(nullptr, c->ts_[Block::kKey]);
  EXPECT_EQ(nullptr, c->ts_[Block::kUKey]);
  EXPECT_NE(nullptr, c->ts_[Block::kIndex]);
  EXPECT_EQ(nullptr, c->ts_[Block::kSIndex]);
  EXPECT_EQ(nullptr, c->ts_[Block::kSSegment]);    
}

}  // io
}  // xdl


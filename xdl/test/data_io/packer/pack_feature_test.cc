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


size_t times = 1;

namespace xdl {
namespace io {


typedef std::map<std::string, std::vector<std::pair<unsigned, float>>> FSMap;
typedef std::map<std::string, std::vector<float>> FDMap;

typedef std::vector<std::pair<FSMap, FDMap>> FTable;

class PackFeatureTest: public ::testing::Test {
  static const size_t kBatchSize;
  static const size_t kSGCount;
  static const size_t kTableCount;
  static const size_t kFeatureCount;

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
  static void InitTable(FeatureTable *ft, FTable &fft, int c, int ktable);
  static void InitSampleGroup(SampleGroup &sg, std::vector<FTable> &fsg, int sgi);
  static void CheckIndicator();
  static void CheckFeature();

  static Device *dev_;
  static Schema schema_;

  static std::vector<std::vector<FTable>> fsgs_;  // [sg][table]
  static std::vector<SampleGroup> sgs_;
};

const size_t PackFeatureTest::kBatchSize = 8192;
//const size_t PackFeatureTest::kBatchSize = 4;
const size_t PackFeatureTest::kSGCount = 128;
//const size_t PackFeatureTest::kSGCount = 2;
const size_t PackFeatureTest::kTableCount = 2;
const size_t PackFeatureTest::kFeatureCount = 16;

Device *PackFeatureTest::dev_ = nullptr;
Schema PackFeatureTest::schema_;

PackFeature *PackFeatureTest::pack_ = nullptr;
Batch *PackFeatureTest::batch_ = nullptr;

std::vector<std::vector<FTable>> PackFeatureTest::fsgs_;
std::vector<SampleGroup> PackFeatureTest::sgs_;

size_t PackFeatureTest::sample_count(size_t ktable) {
  size_t c = kBatchSize/kSGCount - 1;
  for (int k = 0; k < ktable; ++k) {
    c = c / 2 + 1;
  }
  return c;
}

size_t PackFeatureTest::batch_size(size_t ktable) {
    size_t batch_size = kBatchSize;
    if (ktable > 0) {
      size_t c = sample_count(ktable);
      batch_size = c * kSGCount;
      if (sample_count(0) * kSGCount < kBatchSize) {
        batch_size += 1;
      }
    }
    return batch_size;
}

void PackFeatureTest::SetUpTestCase() {
  dev_ = new CpuDevice();
  schema_.batch_size_ = kBatchSize;

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

  fsgs_.resize(kSGCount);
  sgs_.resize(kSGCount);

  std::srand(std::time(nullptr));
  for (int i = 0; i < kSGCount; ++i) {
    InitSampleGroup(sgs_[i], fsgs_[i], i);
    //std::cout << sgs_[i].DebugString() << std::endl;
  }

  pack_ = new PackFeature(dev_, &schema_);
}

void PackFeatureTest::InitTable(FeatureTable *ft, FTable &fft, int c, int ktable) {
  EXPECT_LT(ktable, kTableCount);

  if (times == 1) {
    fft.resize(c);
  }

  for (int i = 0; i < c; ++i) {
    auto fl = ft->add_feature_lines();
    // sparse
    FSMap *smap = nullptr;
    if (times == 1) {
      smap = &fft[i].first;
    }
    int cs = std::rand() % 4 + 1;
    for (int j = 0; j < cs; ++j) {
      std::vector<std::pair<unsigned, float>> kvs;
      auto f = fl->add_features();
      for (int k = 0; k < j+1; ++k) {
        kvs.push_back(std::make_pair(k, 0.1*k));
        auto kv = f->add_values();
        kv->set_key(k);
        kv->set_value(0.1*k);
      }
      if (times == 1) {
        smap->insert(FSMap::value_type(std::to_string(ktable)+"u"+std::to_string(j), kvs));
      }
      f->set_name(std::to_string(ktable)+"u"+std::to_string(j));
      f->set_type(kSparse);
    }

    // dense
    FDMap *dmap = nullptr;
    if (times == 1) {
      dmap = &fft[i].second;
    }
    int cd = std::rand() % 4 + 1;
    for (int j = 0; j < cd; ++j) {
      auto f = fl->add_features();
      auto v = f->add_values();
      std::vector<float> vs;
      for (int k = 0; k < j+1; ++k) {
        vs.push_back(0.1*k);
        v->add_vector(0.1*k);
      }
      if (times == 1) {
        dmap->insert(FDMap::value_type(std::to_string(ktable)+"a"+std::to_string(j), vs));
      }
      f->set_name(std::to_string(ktable)+"a"+std::to_string(j));
      f->set_type(kDense);
    }

    /// refer
    if (ktable < kTableCount - 1) {
      fl->set_refer((i+1)/2);
    }
  }
}

void PackFeatureTest::InitSampleGroup(SampleGroup &sg, std::vector<FTable> &fsgs, int sgi) {
  if (times == 1) {
    fsgs.resize(kTableCount);
  }

  for (int ktable = 0; ktable < kTableCount; ++ktable) {
    auto ft = sg.add_feature_tables();
    int c = sample_count(ktable);
    InitTable(ft, fsgs[ktable], c, ktable);
  }
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
    pparam.begin_ = 0;
    pparam.end_ = sgs_[i].feature_tables(0).feature_lines_size();
    for (int ktable = 0; ktable < kTableCount; ++ktable) {
      pparam.ftable_ = &sgs_[i].feature_tables(ktable);
      pparam.ktable_ = ktable;
      pparam.isgroup_ = i;
      EXPECT_GE(pparam.begin_, 0);
      if (times == 1) {
        EXPECT_LE(pparam.end_, fsgs_[i][ktable].size());
      }
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

  ASSERT_EQ(kFeatureCount*kTableCount*4 + kTableCount-1, batch_->ts_count_);
}

void PackFeatureTest::CheckIndicator() {
  if (times > 1) {
    return;
  }
  ASSERT_EQ(kSGCount, fsgs_.size());
  ASSERT_EQ(kSGCount, sgs_.size());

  if (kTableCount <= 1) {
    return;
  }

  // sampe count accumlated till this sg
  unsigned acc[kTableCount];
  memset(acc, 0, sizeof(acc));
  for (int i = 0; i < kSGCount; ++i) {
    /// for each sample group
    auto sg = sgs_[i];
    for (int t = 0; t < sg.feature_tables_size(); ++t) {
      auto ft = sg.feature_tables(t);

      if (t != sg.feature_tables_size() - 1) {
        auto blk = batch_->Get(kIndicatorPrefix+std::to_string(t));
        auto indices = blk->ts_[Block::kIndex]->Raw<int32_t>();
        for (int c = 0; c < ft.feature_lines_size(); ++c) {
          auto &fl = ft.feature_lines(c);
          ASSERT_EQ(fl.refer()+acc[t+1], indices[acc[t]+c])
              << "i=" << i << " t=" << t << " acc[" << t+1 << "]=" << acc[t+1];
        }
      }

      acc[t] += ft.feature_lines_size();
      //std::cout << "i=" << i << " acc[" << t << "]=" << ft.feature_lines_size() << std::endl;
    }
  }

  // padding
  for (int t = 0; t < kTableCount - 1; ++t) {
    auto blk = batch_->Get(kIndicatorPrefix+std::to_string(t));
    size_t bs = batch_size(t);
    ASSERT_NE(nullptr, blk);
    ASSERT_NE(nullptr, blk->ts_[Block::kIndex]);
    ASSERT_EQ(bs, blk->ts_[Block::kIndex]->Shape()[0]) << "t=" << t 
        << " shape=(" << blk->ts_[Block::kIndex]->Shape()[0] << ") bs=" << bs;
    ASSERT_EQ(1, blk->ts_count_);
    auto indices = blk->ts_[Block::kIndex]->Raw<int32_t>();
    size_t c = sample_count(t) * kSGCount;
    ASSERT_EQ(c, acc[t]);
    for (int i = acc[t]; i < bs; ++i) {
      ASSERT_EQ(acc[t+1], indices[i] + 1) << "t=" << t << " i=" << i << " bs=" << bs;  // TODO: should be ASSERT_EQ(acc[t+1], indices[i])
    }
  }
}

void PackFeatureTest::CheckFeature() {
  if (times > 1) {
    return;
  }
  /// check feature
  for (auto &kv : schema_.feature_opts()) {
    auto &opt = kv.second;
    auto blk = batch_->Get(opt->name());
    ASSERT_NE(nullptr, blk);

    auto value = blk->ts_[Block::kValue];
    auto key = blk->ts_[Block::kKey];
    auto segment = blk->ts_[Block::kSegment];

    float *values = value->Raw<float>();
    //ASSERT_NE(nullptr, values) << "name=" << opt->name() << " table=" << opt->table();

    int64_t *keys = nullptr;
    int32_t *segments = nullptr;

    if (opt->type() == kSparse) {
      ASSERT_NE(nullptr, key);
      keys = key->Raw<int64_t>();
      //ASSERT_NE(nullptr, keys);

      ASSERT_NE(nullptr, segment);
      segments = segment->Raw<int32_t>();
      ASSERT_NE(nullptr, segments);

      EXPECT_EQ(3, blk->ts_count_);
    } else {
      EXPECT_EQ(1, blk->ts_count_);
    }

    auto ktable = opt->table();

    int n = 0;  // sample count
    int m = 0;  // id count

    for (auto fsg: fsgs_) {
      auto &fft = fsg[ktable];
      /// each table
      for (auto &ffl: fft) {
        /// each feature line
        auto &smap = ffl.first;
        auto &dmap = ffl.second;
        if (opt->type() == kSparse) {
          auto it = smap.find(opt->name());
          if (it != smap.end()) {
             auto &vs = it->second;
            for (auto &kv: vs) {
              EXPECT_EQ(kv.first, keys[m*2+1]) << "feature=" << opt->name()
                  << " m=" << m;
              EXPECT_FLOAT_EQ(kv.second, values[m]) << "feature=" << opt->name()
                  << " m=" << m;
              ++m;
            }
          }
          EXPECT_EQ(m, segments[n]) << "feature=" << opt->name()
                  << " m=" << m;
        } else {
          EXPECT_EQ(n*opt->nvec(), m) << "feature=" << opt->name()
                  << " n=" << n << " nvec=" << opt->nvec() << " m=" << m;
          auto it = dmap.find(opt->name());
          if (it != dmap.end()) {
            auto vs = it->second;
            EXPECT_EQ(opt->nvec(), vs.size());
            for (auto v: vs) {
              EXPECT_FLOAT_EQ(v, values[m]) << "feature=" << opt->name()
                  << " m=" << m;
              ++m;
            }
          } else {
            for (int i = 0; i < opt->nvec(); ++i) {
              EXPECT_FLOAT_EQ(0, values[m]) << "feature=" << opt->name()
                  << " m=" << m;
              ++m;
            }
          }
        }
        ++n;
      }  /// for each feature line
    }  /// for each sample group


    int c = sample_count(ktable) * kSGCount;
    ASSERT_EQ(c, n) << "ktable=" << ktable;

    size_t bs = batch_size(ktable);
    /// padding zero
    for (; n < bs; ++n) {
      if (opt->type() == kSparse) {
        EXPECT_EQ(m, segments[n]) << "feature=" << opt->name() << " n=" << n;
      } else {
        for (int i = 0; i < opt->nvec(); ++i) {
          EXPECT_FLOAT_EQ(0, values[m]) << "feature=" << opt->name()
              << " m=" << m;
          ++m;
        }
      }
    }
    ASSERT_EQ(bs, n) << "feature=" << opt->name();

    if (opt->type() == kSparse) {
      auto dims = segment->Shape().Dims();
      EXPECT_EQ(1, dims.size());
      EXPECT_EQ(bs, dims[0]);

      dims = key->Shape().Dims();
      EXPECT_EQ(2, dims.size());
      EXPECT_EQ(m, dims[0]);
      EXPECT_EQ(2, dims[1]);

      dims = value->Shape().Dims();
      EXPECT_EQ(2, dims.size());
      EXPECT_EQ(m, dims[0]);
      EXPECT_EQ(opt->has_nvec()?opt->nvec():1, dims[1]);

      std::cout << "feature=" << opt->name() << " seg" << segment->Shape() << 
          " idx" << key->Shape() << " val" << value->Shape() << " pass check" << std::endl;
      //LOG(INFO) << "feature=" << opt->name() << " seg" << segment->Shape() << 
      //    " idx" << key->Shape() << " val" << value->Shape() << " pass check";
    } else {
      auto dims = value->Shape().Dims();
      EXPECT_EQ(2, dims.size());
      EXPECT_EQ(bs, dims[0]);
      EXPECT_EQ(opt->nvec(), dims[1]);
      //LOG(INFO) << "feature=" << opt->name() << " " << value->Shape() << " pass check";
    }
  }  //feature opt

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

  CheckFeature();

  CheckIndicator();

  EXPECT_EQ(kFeatureCount*kTableCount*4 + kTableCount-1, batch_->ts_count_);
}

TEST_F(PackFeatureTest, Run) {

  std::cout << "run " << times << " times" << std::endl;
  for (size_t i = 0; i < times; ++i) {
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
}

}  // io
}  // xdl

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  for (int i = 1; i < argc; ++i) {
    printf("arg %2d = %s\n", i, argv[i]);
  }

  if (argc > 1) {
    if (isdigit(argv[1][0])) {
      times = atoi(argv[1]);
    }
  }

  return RUN_ALL_TESTS();
}

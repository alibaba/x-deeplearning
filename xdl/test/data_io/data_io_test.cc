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

#include "xdl/data_io/data_io.h"
#include "xdl/data_io/op/debug_op.h"
#include "xdl/data_io/parser/parser.h"
#include "gtest/gtest.h"
#include "xdl/core/utils/logging.h"

#include <string.h>

const char *path = "sample.txt";
size_t epochs = 128;

namespace xdl {
namespace io {

class DataIOTest: public ::testing::Test {
 public:
  static void SetUpTestCase() {
    DataIO *data_io = new DataIO("test", kTxt, kLocal, "");
    EXPECT_NE(nullptr, data_io);

    data_io->SetBatchSize(kBatchSize);
    data_io->SetLabelCount(kLabelCount);

    data_io->AddFeatureOpt("ufav3", kSparse);
    data_io->AddFeatureOpt("upv14", kSparse);
    data_io->AddFeatureOpt("a", kDense, 0, 3);
    data_io->AddFeatureOpt("s", kDense, 0, 1);

    data_io->AddFeatureOpt("item_1", kSparse);
    data_io->AddFeatureOpt("item_2", kSparse);
    data_io->AddFeatureOpt("item_3", kSparse);
    data_io->AddFeatureOpt("item_4", kSparse);
    data_io->AddFeatureOpt("item_5", kSparse);
    data_io->AddFeatureOpt("item_6", kSparse);
    data_io->AddFeatureOpt("item_7", kSparse);
    data_io->AddFeatureOpt("item_8", kSparse);
    data_io->AddFeatureOpt("item_9", kSparse);
    data_io->AddFeatureOpt("item_10", kSparse);

    data_io_ = data_io;
  }

  static void TearDownTestCase() {
  }

  static void TestRun(const char *);

  static const size_t kBatchSize;
  static const size_t kLabelCount;

  static DataIO *data_io_;
};

const size_t DataIOTest::kBatchSize = 16;
const size_t DataIOTest::kLabelCount = 2;


DataIO *DataIOTest::data_io_ = nullptr;

void DataIOTest::TestRun(const char *path) {
  data_io_->AddPath(path);
  data_io_->SetEpochs(epochs);

  //data_io_->AddOp(new DebugOP);

  data_io_->Startup();

  const Batch *batch;
  do {
   batch = data_io_->GetBatch();
   if (batch != nullptr) {
     XDL_CHECK(batch->Get("ufav3") !=nullptr);
     XDL_CHECK(batch->Get("upv14") !=nullptr);
     XDL_CHECK(batch->Get("a") !=nullptr);
     XDL_CHECK(batch->Get("s") !=nullptr);
     XDL_CHECK(batch->Get("item_1") !=nullptr);
     XDL_CHECK(batch->Get("item_2") !=nullptr);
     XDL_CHECK(batch->Get("item_3") !=nullptr);
     XDL_CHECK(batch->Get("item_4") !=nullptr);
     XDL_CHECK(batch->Get("item_5") !=nullptr);
     XDL_CHECK(batch->Get("item_6") !=nullptr);
     XDL_CHECK(batch->Get("item_7") !=nullptr);
     XDL_CHECK(batch->Get("item_8") !=nullptr);
     XDL_CHECK(batch->Get("item_9") !=nullptr);
     XDL_CHECK(batch->Get("item_10") !=nullptr);
     //std::cout << "." << std::endl;
     XDL_CHECK(batch->Get("skey") != nullptr);

     auto blk = batch->Get("skey");
     auto sbuf = blk->ts_[Block::kSBuf];
     XDL_CHECK(sbuf != nullptr);

     auto dims = sbuf->Shape().Dims();
     XDL_CHECK(dims.size() == 2);
     XDL_CHECK(dims[0] == kBatchSize);
     XDL_CHECK(dims[1] > 0);

     size_t max_len = dims[1];

     auto sbufs = sbuf->Raw<int8_t>();

     /*
     for (size_t i = 0; i < kBatchSize; ++i) {
       unsigned len = strlen(&sbufs[i*max_len]);
       XDL_CHECK(len < max_len) << i << " len=" << len << ", max_len=" << max_len;
       char *sk = (char *)&sbufs[i*max_len];
       std::cout << ">>> " << std::string(sk, len) << std::endl;
     }
     */
     std::cout << ". " << std::endl;
   }
  } while(batch != nullptr);

  data_io_->Shutdown();
}

TEST_F(DataIOTest, Run) {
  TestRun(path);
}

}  // namespace io
}  // namespace xdl

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  for (int i = 1; i < argc; ++i) {
    printf("arg %2d = %s\n", i, argv[i]);
  }

  if (argc > 1) {
    if (access(argv[1], 0) == 0) {
      path = strdup(argv[1]);
    }
  }

  if (argc > 2) {
    if (isdigit(argv[2][0])) {
      epochs = atoi(argv[2]);
    }
  }

  return RUN_ALL_TESTS();
}


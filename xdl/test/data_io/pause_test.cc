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
#include "xdl/data_io/op/debug_rebuild_op.h"
#include "xdl/data_io/parser/parser.h"
#include "gtest/gtest.h"
#include "xdl/core/utils/logging.h"

#include <string.h>

const char *path = "sample.txt";
size_t epochs = 1;
size_t pause_count = 1;

namespace xdl {
namespace io {

class PauseTest: public ::testing::Test {
 public:
  static void SetUpTestCase() {
    DataIO *data_io = new DataIO("test", kTxt, kLocal, "");
    EXPECT_NE(nullptr, data_io);

    data_io->SetEpochs(1);
    data_io->SetBatchSize(kBatchSize);
    data_io->SetLabelCount(kLabelCount);

    data_io->AddFeatureOpt("ufav3", kSparse);
    data_io->AddFeatureOpt("upv14", kSparse);
    data_io->AddFeatureOpt("a", kDense, 0, 3);
    data_io->AddFeatureOpt("s", kDense, 0, 1);

    data_io->SetKeepSGroup(true);
    data_io->SetSplitGroup(false);
    data_io->SetPause(pause_count, true);

    data_io->SetThreads(2);

    data_io_ = data_io;
  }

  static void TearDownTestCase() {
    delete data_io_;
    data_io_ = nullptr;
  }

  static void TestRun(const char *);

  static const size_t kBatchSize;
  static const size_t kLabelCount;

  static DataIO *data_io_;
};

const size_t PauseTest::kBatchSize = 2;
const size_t PauseTest::kLabelCount = 2;


DataIO *PauseTest::data_io_ = nullptr;

void PauseTest::TestRun(const char *path) {
  auto op = new DebugRebuildOP();
  std::map<std::string, std::string> kv = {{"limit", std::to_string(pause_count)}};
  op->Init(kv);
  data_io_->AddOp(op);

  data_io_->AddPath(path);
  data_io_->SetEpochs(epochs);

  data_io_->Startup();

  const Batch *batch;
  do {
   batch = data_io_->GetBatch();
   if (batch != nullptr) {
     XDL_CHECK(batch->Get("ufav3") !=nullptr);
     XDL_CHECK(batch->Get("upv14") !=nullptr);
     XDL_CHECK(batch->Get("a") !=nullptr);
     XDL_CHECK(batch->Get("s") !=nullptr);
     std::cerr << ".";
   }
  } while(batch != nullptr);

  std::cerr << std::endl;
}

TEST_F(PauseTest, Run) {
  TestRun(path);
}

}
}

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


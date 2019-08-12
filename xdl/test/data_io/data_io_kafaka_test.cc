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
#include "xdl/data_io/fs/file_system_kafka.h"
#include "gtest/gtest.h"
#include "xdl/core/utils/logging.h"

size_t epochs = 1;

namespace xdl {
namespace io {

class DataIOKafkaTest: public ::testing::Test {
 public:
  static void SetUpTestCase() {
    DataIO *data_io = new DataIO("test_kafka", kTxt, kKafka, "11.139.224.110:9092");
    EXPECT_NE(nullptr, data_io);

    data_io->SetBatchSize(kBatchSize);
    data_io->SetLabelCount(kLabelCount);

    data_io->AddFeatureOpt("ufav3", kSparse);
    data_io->AddFeatureOpt("upv14", kSparse);
    data_io->AddFeatureOpt("a", kDense, 0, 3);
    data_io->AddFeatureOpt("s", kDense, 0, 1);

    data_io_ = data_io;
  }

  static void TearDownTestCase() {
  }

  static void TestDir(void);
  static void TestRun(const char *);
  static void TestKafkaAnt(const char *);

  static const size_t kBatchSize;
  static const size_t kLabelCount;

  static DataIO *data_io_;
};

const size_t DataIOKafkaTest::kBatchSize = 4;
const size_t DataIOKafkaTest::kLabelCount = 2;

DataIO *DataIOKafkaTest::data_io_ = nullptr;

void DataIOKafkaTest::TestDir(void) {
  FileSystem *kfs =FileSystemKafka::Get("11.139.224.110:9092");
  XDL_CHECK(kfs != nullptr);

  /* Kafka dosen't support 'Open' */
  XDL_CHECK(kfs->Open("any_path", "any_mode") == nullptr);

  XDL_CHECK(kfs->IsReg("test:0") == true);
  XDL_CHECK(kfs->IsReg("not_exist") == false);

  /* Kafka doesn't support directory */
  XDL_CHECK(kfs->IsDir("any") == false);

  /* Kafka dosen't support tranverse of dir */
  auto paths = kfs->Dir("any");
  XDL_CHECK(paths.size() == 0);

  XDL_CHECK(kfs->Size("any") == size_t(-1));
}

void DataIOKafkaTest::TestRun(const char *path) {
  data_io_->AddPath(path);
  data_io_->SetEpochs(epochs);

  data_io_->Startup();

  const Batch *batch;
  batch = data_io_->GetBatch();
  XDL_CHECK(nullptr != batch);
  if (batch != nullptr) {
    XDL_CHECK(batch->Get("ufav3") !=nullptr);
    XDL_CHECK(batch->Get("upv14") !=nullptr);
    XDL_CHECK(batch->Get("a") !=nullptr);
    XDL_CHECK(batch->Get("s") !=nullptr);
    std::cout << "." << std::endl;
  }
  data_io_->Shutdown(true);
}

void DataIOKafkaTest::TestKafkaAnt(const char *path) {
  FileSystem *kfs =FileSystemKafka::Get("11.139.224.110:9092");
  XDL_CHECK(kfs != nullptr);

  IOAnt *ant = kfs->GetAnt(path, 'r');
  XDL_CHECK(ant != nullptr);

  /* Kafka dosen't support write for samples */
  XDL_CHECK(ant->Write("any", 3) == ssize_t(-1));

  XDL_CHECK(ant->Seek(0) == 0);
}

TEST_F(DataIOKafkaTest, Run) {
  //TestRun("earth:0");
  TestDir();
  TestKafkaAnt("earth:0");
}

}
}

/*
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
*/


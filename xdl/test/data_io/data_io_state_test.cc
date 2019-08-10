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
#include "xdl/data_io/fs/file_system_local.h"
#include "gtest/gtest.h"
#include "xdl/core/utils/logging.h"

#include <string.h>

const char *path = "sample.txt";
size_t epochs = 3;

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

    data_io->AddPath(path);
    data_io->SetEpochs(epochs);

    data_io_ = data_io;
  }

  static void TearDownTestCase() {
  }

  static void TestState(void);
  static void TestLocal(void);
  static void TestAnt(void);

  static const size_t kBatchSize;
  static const size_t kLabelCount;

  static DataIO *data_io_;
};

void DataIOTest::TestState(void) {
  data_io_->Init();
  std::string text = data_io_->Store();

  std::cout << " >>> " << text << std::endl;

  EXPECT_NE(0, text.size());

  data_io_->Restore(text);
  data_io_->Init();
}

void DataIOTest::TestLocal(void) {
  auto fs = FileSystemLocal::Get();
  XDL_CHECK(fs != nullptr);
  FileSystemLocal *local = reinterpret_cast<FileSystemLocal*>(fs);
  XDL_CHECK(local != nullptr);
  XDL_CHECK(local->IsReg("./data_io_state_test") == true);
  XDL_CHECK(local->IsReg("./not_exist") == false);
  XDL_CHECK(local->IsDir("./data_io_state_test") == false);
  XDL_CHECK(local->IsDir("./") == true);

  auto files = local->Dir("./");
  XDL_CHECK(files.size() > 0);

  void *fp = local->Open("./data_io_state_test", "r");
  XDL_CHECK(fp != nullptr);
}

void DataIOTest::TestAnt(void) {
  auto fs = FileSystemLocal::Get();
  XDL_CHECK(fs != nullptr);

  auto ant = fs->GetAnt("./local_test.txt", 'w');
  XDL_CHECK(ant != nullptr);

  std::string content = "Fire, earth, storm. Hear my call!";
  ssize_t res = ant->Write(content.c_str(), content.length());
  XDL_CHECK(res == content.length());

  delete ant; /* Deliberately close file */

  fs = FileSystemLocal::Get();
  XDL_CHECK(fs != nullptr);

  ant = fs->GetAnt("./local_test.txt", 'r');
  char buff[1024];
  res = ant->Read(buff, content.length());
  XDL_CHECK(res == content.length());

  XDL_CHECK(ant->Seek(3) == 0);
}

const size_t DataIOTest::kBatchSize = 16;
const size_t DataIOTest::kLabelCount = 2;

DataIO *DataIOTest::data_io_ = nullptr;

TEST_F(DataIOTest, Run) {
  TestState();
  TestLocal();
  TestAnt();
}

}  // namespace io
}  // namespace xdl

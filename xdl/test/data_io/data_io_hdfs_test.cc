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
#include "test/util/hdfs_launcher.h"
#include "xdl/data_io/parser/parser.h"
#include "xdl/data_io/fs/file_system_hdfs.h"
#include "gtest/gtest.h"

#include <string.h>

const char *path = "hdfs://127.0.0.1:9090/test_data/data_io/sample.txt";
const char *dir = "hdfs://127.0.0.1:9090/test_data/data_io/";
size_t epochs = 1;

namespace xdl {
namespace io {

class DataIOTest: public ::testing::Test {
 public:
  void SetUp() override {
    if (false == ps::HDFSLauncher::Start()) {
      skip_ = true;
    }
    if (skip_) {
      GTEST_SKIP();
      return;
    }

    DataIO *data_io = new DataIO("test", kTxt, kHdfs, "hdfs://127.0.0.1:9090");
    EXPECT_NE(nullptr, data_io);

    data_io->SetEpochs(10);
    data_io->SetBatchSize(kBatchSize);
    data_io->SetLabelCount(kLabelCount);

    data_io->AddFeatureOpt("ufav3", kSparse);
    data_io->AddFeatureOpt("upv14", kSparse);
    data_io->AddFeatureOpt("a", kDense, 0, 3);
    data_io->AddFeatureOpt("s", kDense, 0, 1);

    data_io_ = data_io;
  }

  void TearDown() override {
    if (!skip_) {
      ps::HDFSLauncher::Stop();
    }
  }

  static void TestRun(const char *);
  static void TestHdfs(const char *);
  static void TestAnt(const char *);

  static const size_t kBatchSize;
  static const size_t kLabelCount;

  static DataIO *data_io_;

private:
  bool skip_ = false;
};

const size_t DataIOTest::kBatchSize = 1024;
const size_t DataIOTest::kLabelCount = 2;


DataIO *DataIOTest::data_io_ = nullptr;

void DataIOTest::TestRun(const char *path) {
  //data_io_->SetMeta(meta);
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
     std::cerr << ".";
   }
  } while(batch != nullptr);

  std::cerr << std::endl;

  data_io_->Shutdown();
}

void DataIOTest::TestHdfs(const char *path) {
  auto fs = FileSystemHdfs::Get("hdfs://127.0.0.1:9090");
  XDL_CHECK(fs != nullptr);
  FileSystemHdfs *hdfs = reinterpret_cast<FileSystemHdfs*>(fs);
  XDL_CHECK(hdfs != nullptr);
  XDL_CHECK(hdfs->IsReg(path) == true);
  XDL_CHECK(hdfs->IsDir(path) == false);

  auto files = hdfs->Dir(dir);
  XDL_CHECK(files.size() > 1);

  size_t st = hdfs->Size(path);
  XDL_CHECK(st > 1);

  void *fd = hdfs->Open(path, "r");
  XDL_CHECK(fd != nullptr);

  std::string node = hdfs->Path2Node("ftp://192.168.0.1");
  XDL_CHECK(node == "");
  node = hdfs->Path2Node("hdfs://192.168.0.1/hello");
  XDL_CHECK(node == "hdfs://192.168.0.1");

  std::string content = "What is nature's call?";
  std::string file_name = std::string(dir) + "/write.test";
  HdfsWrite(file_name, content);
  XDL_CHECK(HdfsRead(file_name) == content);
}

void DataIOTest::TestAnt(const char *path) {
  auto fs = GetHdfsFileSystem(path);
  XDL_CHECK(fs != nullptr);

  std::string file_name = std::string(dir) + "/ioant.test";
  std::string content = "Fire, earth, storm. Hear my call!";
  auto ant = fs->GetAnt(file_name.c_str(), 'w');
  XDL_CHECK(ant != nullptr);

  ssize_t res = ant->Write(content.c_str(), content.length());
  XDL_CHECK(res == content.length());

  delete ant; /* Need it to sync hdfs ! */

  fs = GetHdfsFileSystem(path);
  XDL_CHECK(fs != nullptr);

  ant = fs->GetAnt(file_name.c_str(), 'r');
  char buff[1024];
  res = ant->Read(buff, content.length());
  XDL_CHECK(res == content.length());
}

TEST_F(DataIOTest, Run) {
  TestRun(path);
}

TEST_F(DataIOTest, Hdfs) {
  TestHdfs(path);
}

TEST_F(DataIOTest, Ant) {
  TestAnt(path);
}

}
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


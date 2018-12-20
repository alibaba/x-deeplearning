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

#include "xdl/data_io/fs/file_system.h"
#include "test/util/hdfs_launcher.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <iostream>

namespace xdl {
namespace io {

const char *path = "hdfs://127.0.0.1:9090/test_data/data_io";

class HdfsTest : public testing::Test {
  public:
    void SetUp() override {
      if (false == ps::HDFSLauncher::Start()) {
        skip_ = true;
      }
      if (skip_) {
        GTEST_SKIP();
      }
    }

    void TearDown() override {
      if (!skip_) {
        ps::HDFSLauncher::Stop();
      }
    }

  private:
    bool skip_ = false;
};

TEST_F(HdfsTest, TestHdfs) {
  auto fs = GetFileSystem(kHdfs, "hdfs://127.0.0.1:9090");
  std::cout << "connect" << std::endl;

  auto ret = fs->IsDir(path);
  ASSERT_TRUE(ret);

  auto paths = fs->Dir("hdfs://127.0.0.1:9090/test_data/data_io");
  
  EXPECT_GT(paths.size(), 0);
  for (auto &path: paths) {
    std::cout << path << std::endl;
  }

  auto size = fs->Size("hdfs://127.0.0.1:9090/test_data/data_io/tdm.dat");
  std::cout << "size=" << size << std::endl;
}

}
}

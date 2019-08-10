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
#include "xdl/data_io/fs/zlib_ant.h"
#include "xdl/data_io/fs/file_system_local.h"
#include "gtest/gtest.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

class FSZlibTest: public ::testing::Test {
 public:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }

  static void TestZlibAnt(const char *);
};

void FSZlibTest::TestZlibAnt(const char *path) {
  auto fs = FileSystemLocal::Get();
  XDL_CHECK(fs != nullptr);
  // content is "abcdefghijklmnopqrstuvwxyz1234567890" * 10
  IOAnt* local_ant = fs->GetAnt("./zlib_test_data", 'r');
  XDL_CHECK(local_ant != nullptr);
  //ZlibCompressionOptions options = ZlibCompressionOptions::DEFAULT();
  //IOAnt* ant = new ZlibAnt(local_ant, options);
  IOAnt* ant = new ZlibAnt(local_ant);  

  char buff[1024];
  memset(buff, 0, sizeof(buff));
  const char* str = "abcdefghijklmnopqrstuvwxyz1234567890";
  for (size_t i = 0; i < 9; i++) {
    ssize_t ret = ant->Read(buff, 36);
    EXPECT_EQ(ret, 36);
    EXPECT_STREQ(str, buff);
  }
  ssize_t last = ant->Read(buff, 100);
  EXPECT_EQ(last, 36);
  EXPECT_STREQ(str, buff);

  //SEEK to begin
  XDL_CHECK(ant->Seek(0) == 0);
  ssize_t ret = ant->Read(buff, 1000);
  EXPECT_EQ(ret, 360);
  char tmp[1024];
  for (size_t i = 0; i < 10; i++) {
    strncpy(tmp, &buff[i*36], 36);
    tmp[36] = 0;
    EXPECT_STREQ(str, tmp);
  }
  EXPECT_EQ(ant->Read(buff, 1000), 0);

  //SEEK to spec
  memset(buff, 0, sizeof(buff));
  XDL_CHECK(ant->Seek(36*9) == 36*9);
  ret = ant->Read(buff, 1000);
  EXPECT_EQ(ret, 36);
  EXPECT_STREQ(str, buff);
  EXPECT_EQ(ant->Read(buff, 1000), 0);
}

TEST_F(FSZlibTest, Run) {
  TestZlibAnt("earth:0");
}

}
}

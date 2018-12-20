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

#include "gtest/gtest.h"
#include "xdl/core/utils/file_utils.h"

using xdl::FileUtils;

TEST(FileTest, File) {

  {
    std::string cont = FileUtils::ReadLocalFile("./not_exist.txt");
    ASSERT_EQ(cont, "");

    cont = FileUtils::ReadLocalBinaryFile("./not_exist.bin");
    ASSERT_EQ(cont, "");
  }

  {
    std::string content = "We are our desires";
    bool res = FileUtils::WriteLocalFile("./file_test.txt", content);
    ASSERT_EQ(res, true);

    res = FileUtils::MoveFile("./file_test.txt", "./new_test.txt");
    ASSERT_EQ(res, true);

    res = FileUtils::DeleteLocalFile("./new_test.txt");
    ASSERT_EQ(res, true);
  }
  
  {
    std::string content = "Open the door";
    bool res = FileUtils::WriteLocalFile("./a.txt", content);
    ASSERT_EQ(res, true);
    res = FileUtils::WriteLocalFile("./c.txt", content);
    ASSERT_EQ(res, true);

    content = "Close that gate";
    res = FileUtils::WriteLocalFile("./b.txt", content);
    ASSERT_EQ(res, true);

    res = FileUtils::CompFile(std::string("./a.txt"), std::string("./b.txt"));
    ASSERT_EQ(res, false);
    res = FileUtils::CompFile(std::string("./a.txt"), std::string("./c.txt"));
    ASSERT_EQ(res, true);

    int ret = FileUtils::CompFile("./a.txt", "./b.txt");
    ASSERT_EQ(ret, 1);
    ret = FileUtils::CompFile("./a.txt", "./c.txt");
    ASSERT_EQ(ret, 0);
  }

  {
    bool res = FileUtils::IsDirExists("./");
    ASSERT_EQ(res, true);
    res = FileUtils::IsDirExists("./not_exists/");
    ASSERT_EQ(res, false);

    res = FileUtils::CreatDir("./exists/");
    ASSERT_EQ(res, true);
    res = FileUtils::IsDirExists("./exists/");
    ASSERT_EQ(res, true);
  }

  {
    bool res = FileUtils::TouchFile("./touch_it.txt");
    ASSERT_EQ(res, true);

    res = FileUtils::CopyFile("./touch_it.txt", "./copy_it.txt");
    ASSERT_EQ(res, true);

    res = FileUtils::IsFileExist("./copy_it.txt");
    ASSERT_EQ(res, true);

    size_t st = FileUtils::FileSize("./copy_it.txt");
    ASSERT_EQ(st, 0);
  }
}

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
#include "test/util/hdfs_launcher.h"
#include "ps-plus/common/file_system.h"

using ps::FileSystem;
using ps::Status;

class FileSystemTest : public testing::Test {
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

TEST_F(FileSystemTest, FileSystem) {
  {
    FileSystem::WriteStream *stream = nullptr;
    FileSystem::OpenWriteStreamAny("./ows.txt",  &stream, false);
    ASSERT_NE(stream, nullptr);
  }

  {
    FileSystem::ReadStream *stream = nullptr;
    FileSystem::OpenReadStreamAny("./ows.txt",  &stream);
    ASSERT_NE(stream, nullptr);
  }

  {
    Status st = FileSystem::MkdirAny("./any");
    ASSERT_TRUE(st.IsOk());

    std::vector<std::string> files;
    st = FileSystem::ListDirectoryAny("./any", &files);
    ASSERT_TRUE(st.IsOk());
    ASSERT_EQ(files.size(), 0);
  }

  {
    FileSystem::WriteStream *stream = nullptr;
    FileSystem::OpenWriteStreamAny("./os.txt", &stream, false);
    ASSERT_NE(stream, nullptr);
    delete stream;

    Status st = FileSystem::RenameAny("./os.txt", "./so.txt");
    ASSERT_TRUE(st.IsOk());

    st = FileSystem::RemoveAny("./so.txt");
    ASSERT_TRUE(st.IsOk());
  }

  { /* Test MemoryFileSystem */
    auto fs = ps::GetPlugin<FileSystem>("memory");
    ASSERT_NE(fs, nullptr);
    Status st = fs->Mkdir("./Wednesday");
    ASSERT_EQ(st, Status::Ok());

    std::vector<std::string> files;
    st = fs->ListDirectory("./Wednesday", &files);
    ASSERT_EQ(st, Status::Ok());
    ASSERT_EQ(files.size(), 0);

    FileSystem::WriteStream *stream = nullptr;
    FileSystem::OpenWriteStreamAny("./Wed.txt", &stream, false);
    ASSERT_NE(stream, nullptr);
    delete stream;

    st = fs->Remove("./Wed.txt");
    ASSERT_EQ(st, Status::Ok());

    st = fs->Rename("./Wed.txt", "./Thurs.txt");
    ASSERT_EQ(st, Status::Ok());
  }

  { /* Test FileFileSystem */
    auto fs = ps::GetPlugin<FileSystem>("file");
    FileSystem::WriteStream *ws = nullptr;
    Status st = fs->OpenWriteStream("ffs.txt", &ws);
    ASSERT_EQ(st, Status::Ok());
    ASSERT_NE(ws, nullptr);

    int64_t sz = ws->WriteSimple("hello", 5);
    ASSERT_EQ(sz, 5);
    ws->Flush();

    FileSystem::ReadStream *rs = nullptr;
    st = fs->OpenReadStream("ffs.txt", &rs);
    ASSERT_EQ(st, Status::Ok());
    ASSERT_NE(rs, nullptr);
    char buf[32];
    sz = rs->ReadSimple(buf, sizeof(buf));
    ASSERT_EQ(sz, 5);
    delete rs;
  }

  { /* Test NoneFileSystem */
    auto fs = ps::GetPlugin<FileSystem>("none");
    FileSystem::WriteStream *ws = nullptr;
    Status st = fs->OpenWriteStream("nfs.txt", &ws);
    ASSERT_EQ(st, Status::Ok());
    ASSERT_NE(ws, nullptr);

    FileSystem::ReadStream *rs = nullptr;
    st = fs->OpenReadStream("ffs.txt", &rs);
    ASSERT_NE(st, Status::Ok());
    ASSERT_EQ(rs, nullptr);

    st = fs->Mkdir("hello");
    ASSERT_EQ(st, Status::Ok());
    st = fs->ListDirectory("hello", nullptr);
    ASSERT_EQ(st, Status::Ok());
    st = fs->Remove("hello");
    ASSERT_EQ(st, Status::Ok());
    st = fs->Rename("hello", "world");
    ASSERT_EQ(st, Status::Ok());
  }

  { /* Test HdfsFileSystem */
    auto fs = ps::GetPlugin<FileSystem>("hdfs");
    FileSystem::WriteStream *ws = nullptr;
    Status st = fs->OpenWriteStream("hdfs://127.0.0.1:9090/hfs.txt", &ws);
    ASSERT_EQ(st, Status::Ok());
    ASSERT_NE(ws, nullptr);

    int64_t sz = ws->WriteSimple("hello", 5);
    ASSERT_EQ(sz, 5);
    ws->Flush();
    delete ws;

    FileSystem::ReadStream *rs = nullptr;
    st = fs->OpenReadStream("hdfs://127.0.0.1:9090/hfs.txt", &rs);
    ASSERT_EQ(st, Status::Ok());
    ASSERT_NE(rs, nullptr);
    char buf[32];
    sz = rs->ReadSimple(buf, sizeof(buf));
    ASSERT_EQ(sz, 5);
    delete rs;

    st = fs->Mkdir("test_dir");
    ASSERT_NE(st, Status::Ok());

    std::vector<std::string> files;
    fs->ListDirectory("test_dir", &files);
    ASSERT_EQ(files.size(), 0);

    st = fs->Remove("not_exist");
    ASSERT_NE(st, Status::Ok());

    st = fs->Rename("left", "right");
    ASSERT_NE(st, Status::Ok());
  }

}

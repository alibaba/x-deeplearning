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
#include "ps-plus/common/data_source.h"
#include "ps-plus/common/hdfs_data_source.h"
#include "test/util/hdfs_launcher.h"

using ps::Status;
using ps::DataClosure;
using ps::HdfsDataSource;

class MockHdfsDS : public HdfsDataSource {
public:
  MockHdfsDS(const std::string &file_path, size_t file_num) : HdfsDataSource(file_path, file_num) {}
  Status Call(const std::string &stream) {
    return InitFromFile(stream);
  }
};

class HdfsDataSourceTest : public testing::Test {
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

TEST_F(HdfsDataSourceTest, HdfsDataSource) {
  {
    std::unique_ptr<HdfsDataSource> hds(new HdfsDataSource("hdfs://127.0.0.1:9090/test_data/data_io/", 1));
    ASSERT_NE(hds, nullptr);

    Status st = hds->Init(1, 1, 100);
    ASSERT_EQ(st, Status::Ok());

    DataClosure dc;
    st = hds->Get(1, &dc);
    ASSERT_NE(st, Status::Ok());

    std::vector<int64_t> ids;
    ids.push_back(1);
    ids.push_back(2);
    std::vector<DataClosure> cs;
    st = hds->BatchGet(ids, &cs);
    ASSERT_EQ(cs.size(), 1);

    std::vector<int64_t> rst;
    hds->BatchGetV2(ids, &cs, &rst);
    ASSERT_EQ(cs.size(), 0);
  }

  {
    auto ds = new MockHdfsDS("hdfs://127.0.0.1:9090/test_data/data_io/", 1);
    Status st = ds->Call("hdfs://127.0.0.1:9090/test_data/data_io/");
    ASSERT_NE(st, Status::Ok());
    delete ds;
  }
}

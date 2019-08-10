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
  static void SetUpTestCase() {}
  static void TearDownTestCase() {}
};

TEST_F(HdfsDataSourceTest, HdfsDataSource) {
  int hdfs_port = xdl::HDFSLauncher::Instance()->GetPort();
  std::string hdfs_prefix = "hdfs://127.0.0.1:" + std::to_string(hdfs_port);
  std::string dir = hdfs_prefix + "/test_data/data_io/";
  {
    std::unique_ptr<HdfsDataSource> hds(new HdfsDataSource(dir, 1));
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
    auto ds = new MockHdfsDS(dir, 1);
    Status st = ds->Call(dir);
    ASSERT_NE(st, Status::Ok());
    delete ds;
  }
}

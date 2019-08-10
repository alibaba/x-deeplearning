#include "gtest/gtest.h"
#include "ps-plus/common/zk_wrapper.h"
#include "test/util/zookeeper_launcher.h"

using ps::ZkWrapper;

class ZkWrapperTest : public testing::Test {
  public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

void call_back(ZkWrapper *zw, const std::string & msg, ZkWrapper::ZkStatus) {
  std::cout << msg << "\n";
}

TEST_F(ZkWrapperTest, ZkWrapper) {
  int zk_port = xdl::ZookeeperLauncher::Instance()->GetPort();
  std::string dir = "127.0.0.1:" + std::to_string(zk_port);
  {
    auto zk = new ZkWrapper(dir, 1000);
    ASSERT_NE(zk, nullptr);

    zk->SetConnCallback(call_back);
    zk->SetChildCallback(call_back);
    zk->SetDataCallback(call_back);
    zk->SetCreateCallback(call_back);
    zk->SetDeleteCallback(call_back);

    zk->Open();

    bool res = zk->Touch("", "", false);
    ASSERT_EQ(res, false);

    zk->Remove("/hello");
    res = zk->Touch("/hello", "world", false);
    ASSERT_EQ(res, true);

    zk->Remove("/robin");
    res = zk->CreatePath("/robin");
    ASSERT_EQ(res, true);

    std::string result;
    res = zk->TouchSeq("/earth", result, "storm", false);
    ASSERT_EQ(res, true);
    std::cout << "result:" << result << "\n";

    std::vector<std::string> files;
    res = zk->GetChild("/robin", files, false);
    ASSERT_EQ(res, true);
    ASSERT_EQ(files.size(), 0);

    std::string content;
    res = zk->GetData("/hello", content, false);
    ASSERT_EQ(res, true);
    ASSERT_EQ(content, "world");

    bool exist = false;
    res = zk->Check("/hello", exist, false);
    ASSERT_EQ(res, true);
    ASSERT_EQ(exist, true);

    int ret = zk->GetState();
    ASSERT_NE(ret, ZOO_EXPIRED_SESSION_STATE);

    ASSERT_TRUE(zk->IsConnected());
    ASSERT_FALSE(zk->IsConnecting());

    ASSERT_FALSE(zk->IsBad());
    ASSERT_NE(zk->GetStatus(), ZkWrapper::ZK_BAD);

    zk->Close();
    delete zk;
  }
}

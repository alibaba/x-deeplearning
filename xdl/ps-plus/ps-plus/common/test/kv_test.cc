#include "gtest/gtest.h"
#include "ps-plus/common/reliable_kv.h"
#include "test/util/zookeeper_launcher.h"

using ps::ReliableKV;
using ps::Status;

class ReliableKVTest : public testing::Test {
  public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(ReliableKVTest, Reliable) {
  int zk_port = xdl::ZookeeperLauncher::Instance()->GetPort();
  std::string zk_prefix = "zfs://127.0.0.1:" + std::to_string(zk_port);
  std::string dir = zk_prefix + "/";
  std::cout << dir + "kva.test" << "\n";
  {
    auto kv = ps::GetPlugin<ReliableKV>("zfs");
    ASSERT_NE(kv, nullptr);

    Status st = kv->WriteAny(dir + "kva.test", "waterfall", 1);
    ASSERT_EQ(st, Status::Ok());

    
    std::string value;
    st = kv->ReadAny(dir + "kva.test", &value, 1);
    ASSERT_EQ(st, Status::Ok());
    ASSERT_EQ(value, "waterfall");
  }

  {
    auto kv = ps::GetPlugin<ReliableKV>("zfs");
    ASSERT_NE(kv, nullptr);

    Status st = kv->Write(dir + "kv.test", "waterfall", 1);
    ASSERT_EQ(st, Status::Ok());

    
    std::string value;
    st = kv->Read(dir + "kv.test", &value, 1);
    ASSERT_EQ(st, Status::Ok());
    ASSERT_EQ(value, "waterfall");
  }
}

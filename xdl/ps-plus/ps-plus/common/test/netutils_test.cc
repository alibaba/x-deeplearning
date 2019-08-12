#include "gtest/gtest.h"
#include "ps-plus/common/net_utils.h"

using ps::NetUtils;

TEST(NetUtilsTest, NetUtils) {
  std::string hostname;
  bool res = NetUtils::GetHostName(hostname);
  ASSERT_EQ(res, true);
  ASSERT_TRUE(hostname.length() > 0);

  std::vector<std::string> ips;
  res = NetUtils::GetIP(ips);
  ASSERT_EQ(res, true);
  ASSERT_TRUE(ips.size() > 0);

  std::string ip;
  res = NetUtils::GetDefaultIP(ip);
  ASSERT_EQ(res, true);
  ASSERT_TRUE(ip.length() > 0);

  ip = NetUtils::GetLocalIP("bond0");
  ASSERT_TRUE(ip.length() > 0);

  int port = NetUtils::GetAvailablePort();
  ASSERT_TRUE(port > 0);

  int cpu_num = NetUtils::GetAvailableCpuNum();
  ASSERT_TRUE(cpu_num > 0);

  std::string value = NetUtils::GetEnv("NOTEXISTS");
  ASSERT_TRUE(value.length() == 0);
}

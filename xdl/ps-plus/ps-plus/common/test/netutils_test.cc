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

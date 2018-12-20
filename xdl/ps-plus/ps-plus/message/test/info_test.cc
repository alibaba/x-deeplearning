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
#include "ps-plus/message/server_info.h"
#include "ps-plus/message/cluster_info.h"

using ps::ServerInfo;
using ps::ClusterInfo;

TEST(ServerInfoTest, ServerInfo) {
  auto si = new ServerInfo(0, 123, 456, "10.97.192.24", 8080);
  ASSERT_NE(si, nullptr);

  std::string addr = si->Address();
  ASSERT_EQ(addr, "10.97.192.24:8080");

  std::string str = si->ToString();
  ASSERT_EQ(str, "0-123 (10.97.192.24:8080@456)");

  auto asi = new ServerInfo(0, 321, 789, "10.97.192.25", 8080);
  ASSERT_TRUE(asi != si);

  auto ci = new ClusterInfo();
  ASSERT_NE(ci, nullptr);
  ci->AddServer(*si);
  ci->AddServer(*asi);
  ASSERT_EQ(ci->servers_.size(), 2);
}


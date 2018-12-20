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
#include "ps-plus/common/zk_wrapper.h"
#include "test/util/zookeeper_launcher.h"

using ps::ZkWrapper;

class ZkWrapperTest : public testing::Test {
  public:
    static void SetUpTestCase() {
      ps::ZookeeperLauncher::Start();
    }

    static void TearDownTestCase() {
      ps::ZookeeperLauncher::Stop();
    }
};

void call_back(ZkWrapper *zw, const std::string & msg, ZkWrapper::ZkStatus) {
  std::cout << msg << "\n";
}

TEST_F(ZkWrapperTest, ZkWrapper) {
  {
    auto zk = new ZkWrapper("127.0.0.1:2181", 1000);
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

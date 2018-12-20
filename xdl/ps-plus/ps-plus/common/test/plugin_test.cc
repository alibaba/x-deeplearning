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
#include "ps-plus/common/plugin.h"

class MockPlugin {
public:
  virtual ~MockPlugin() {}
  virtual int id() = 0;
};

class MockPlugin1 : public MockPlugin {
public:
  virtual int id() override { return 1; }
};

class MockPluginX : public MockPlugin {
public:
  MockPluginX(int x) : x_(x) {}
  virtual int id() override { return x_; }
private:
  int x_;
};

PLUGIN_REGISTER(MockPlugin, MockPlugin1, MockPlugin1);
PLUGIN_REGISTER(MockPlugin, MockPlugin2, MockPluginX, 2);
PLUGIN_REGISTER(MockPlugin, MockPlugin3, MockPluginX, 3);

TEST(PluginTest, GetPlugin) {
  EXPECT_NE(nullptr, ps::GetPlugin<MockPlugin>("MockPlugin1"));
  EXPECT_NE(nullptr, ps::GetPlugin<MockPlugin>("MockPlugin2"));
  EXPECT_NE(nullptr, ps::GetPlugin<MockPlugin>("MockPlugin3"));
  EXPECT_EQ(nullptr, ps::GetPlugin<MockPlugin>("MockPlugin4"));
  EXPECT_EQ(1, ps::GetPlugin<MockPlugin>("MockPlugin1")->id());
  EXPECT_EQ(2, ps::GetPlugin<MockPlugin>("MockPlugin2")->id());
  EXPECT_EQ(3, ps::GetPlugin<MockPlugin>("MockPlugin3")->id());
}

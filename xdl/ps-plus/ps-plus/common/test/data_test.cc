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
#include "ps-plus/common/data.h"

using ps::Data;
using ps::WrapperData;

namespace {

class MockData {
 public:
  MockData() {
  }
  MockData(int x) {
    init_type = 0;
  }
  MockData(const MockData&) {
    init_type = 1;
  }
  MockData(MockData&&) {
    init_type = 2;
  }
  int init_type;
};

}

TEST(DataTest, WrapperData) {
  MockData d;
  WrapperData<MockData>* d1 = new WrapperData<MockData>(1);
  WrapperData<MockData>* d2 = new WrapperData<MockData>(d);
  WrapperData<MockData>* d3 = new WrapperData<MockData>(std::move(d1->Internal()));
  EXPECT_EQ(0, d1->Internal().init_type);
  EXPECT_EQ(1, d2->Internal().init_type);
  EXPECT_EQ(2, d3->Internal().init_type);
}


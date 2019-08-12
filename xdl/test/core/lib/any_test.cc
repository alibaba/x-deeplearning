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

/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "xdl/core/lib/any.h"
#include "gtest/gtest.h"

namespace xdl {

TEST(AnyTest, TestAll) {
  {
    Any a = std::string("any");
    ASSERT_TRUE(!a.Empty());
    ASSERT_EQ(std::string("any"), a.AnyCast<std::string>());
  }

  {
    Any a = 1;
    ASSERT_TRUE(!a.Empty());
    ASSERT_EQ(1, a.AnyCast<int>());
  }

  {
    Any a;
    Any b = std::string("copy");
    a = b;
    ASSERT_EQ(std::string("copy"), a.AnyCast<std::string>());
  }
}

}  // namespace xdl

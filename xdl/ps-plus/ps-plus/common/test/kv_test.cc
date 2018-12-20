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
#include "ps-plus/common/reliable_kv.h"

using ps::ReliableKV;
using ps::Status;

TEST(ReliableKVTest, Reliable) {
  // {
  //   auto kv = ps::GetPlugin<ReliableKV>("zfs");
  //   ASSERT_NE(kv, nullptr);

  //   Status st = kv->WriteAny("zfs://127.0.0.1:2181/kva.test", "waterfall", 1);
  //   ASSERT_EQ(st, Status::Ok());

    
  //   std::string value;
  //   st = kv->ReadAny("zfs://127.0.0.1:2181/kva.test", &value, 1);
  //   ASSERT_EQ(st, Status::Ok());
  //   ASSERT_EQ(value, "waterfall");
  // }

  // {
  //   auto kv = ps::GetPlugin<ReliableKV>("zfs");
  //   ASSERT_NE(kv, nullptr);

  //   Status st = kv->Write("zfs://127.0.0.1:2181/kv.test", "waterfall", 1);
  //   ASSERT_EQ(st, Status::Ok());

    
  //   std::string value;
  //   st = kv->Read("zfs://127.0.0.1:2181/kv.test", &value, 1);
  //   ASSERT_EQ(st, Status::Ok());
  //   ASSERT_EQ(value, "waterfall");
  // }
}

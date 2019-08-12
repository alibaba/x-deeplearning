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
#include "ps-plus/common/murmurhash.h"
#include <iomanip>
#include <sstream>

using ps::MurmurHash;

std::string uint128_2_string(uint64_t res[2]) {
  std::ostringstream oss;
  oss << std::setw(16) << std::setfill('0') << std::hex << res[0];
  oss << std::setw(16) << std::setfill('0') << std::hex << res[1];
  return oss.str();
}

TEST(MurmurHashTest, MurmurHash) {
  uint64_t res[2];
  // default seed
  {
    MurmurHash hash_fn(0);
    std::string key0("");
    std::string val0("00000000000000000000000000000000");
    std::string key1("this is a test line");
    std::string val1("097eec3aaefa7332e0d14cf2d59d26c3");
    hash_fn(key0.c_str(), key0.size(), res);
    ASSERT_EQ(uint128_2_string(res), val0);
    hash_fn(key1.c_str(), key1.size(), res);
    ASSERT_EQ(uint128_2_string(res), val1);
  }
  
  // use seed
  {
    MurmurHash hash_fn(7621U);
    std::string key0("");
    std::string val0("2087cc3f33e14ed78d8fdd6d054f1c2c");
    std::string key1("this is a test line");
    std::string val1("b3c8e4d2b3806183bc117d2cd9cf9a3e");
    hash_fn(key0.c_str(), key0.size(), res);
    ASSERT_EQ(uint128_2_string(res), val0);
    hash_fn(key1.c_str(), key1.size(), res);
    ASSERT_EQ(uint128_2_string(res), val1);
  }
}

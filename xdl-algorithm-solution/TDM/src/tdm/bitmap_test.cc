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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#include "tdm/bitmap.h"

#include "gtest/gtest.h"

namespace tdm {

TEST(Bitmap, TestConstructor) {
  Bitmap bitmap;
  ASSERT_FALSE(bitmap.is_filtered(100));
  ASSERT_TRUE(bitmap.save("bitmap.bm"));
}

TEST(Bitmap, TestSetFilter) {
  Bitmap bitmap("bitmap.bm");

  size_t index = 100;
  ASSERT_TRUE(bitmap.set_filter(index, true));
  ASSERT_TRUE(bitmap.is_filtered(index));

  index = 1000;
  ASSERT_TRUE(bitmap.set_filter(index, true));
  ASSERT_TRUE(bitmap.is_filtered(index));

  index = 2000;
  ASSERT_TRUE(bitmap.set_filter(index, true));
  ASSERT_TRUE(bitmap.is_filtered(index));

  index = 5000;
  ASSERT_TRUE(bitmap.set_filter(index, true));
  ASSERT_TRUE(bitmap.is_filtered(index));

  ASSERT_TRUE(bitmap.save("bitmap.bm"));
}

TEST(Bitmap, TestSave) {
  Bitmap bitmap("bitmap.bm");
  ASSERT_TRUE(bitmap.is_filtered(5000));
}

}  // namespace tdm

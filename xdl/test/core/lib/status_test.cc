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
#include "xdl/core/lib/status.h"

using xdl::Status;

TEST(StatusTest, DefaultConstructor) {
  Status st;
  ASSERT_EQ(Status::kOk, st.Code());
  ASSERT_EQ("", st.Msg());
}

TEST(StatusTest, CopyAndMove) {
  Status st1 = Status::Ok();
  ASSERT_EQ(Status::kOk, st1.Code());
  ASSERT_EQ("", st1.Msg());

  Status st2 = Status::IndexOverflow("my error");
  ASSERT_EQ(Status::kIndexOverflow, st2.Code());
  ASSERT_EQ("my error", st2.Msg());

  Status st3;
  st3 = st1;
  ASSERT_EQ(Status::kOk, st3.Code());
  ASSERT_EQ("", st3.Msg());

  st3 = st2;
  ASSERT_EQ(Status::kIndexOverflow, st3.Code());
  ASSERT_EQ("my error", st3.Msg());

  Status st4;
  st4 = std::move(st1);
  ASSERT_EQ(Status::kOk, st4.Code());
  ASSERT_EQ("", st4.Msg());

  Status st5;
  st5 = std::move(st2);
  ASSERT_EQ(Status::kIndexOverflow, st5.Code());
  ASSERT_EQ("my error", st5.Msg());

  Status st6;
  st6 = std::move(st5);
  ASSERT_EQ(Status::kIndexOverflow, st6.Code());
  ASSERT_EQ("my error", st6.Msg());

  Status st7(st4);
  ASSERT_EQ(Status::kOk, st7.Code());
  ASSERT_EQ("", st7.Msg());

  Status st8(std::move(st7));
  ASSERT_EQ(Status::kOk, st8.Code());
  ASSERT_EQ("", st8.Msg());

  Status st9(st6);
  ASSERT_EQ(Status::kIndexOverflow, st9.Code());
  ASSERT_EQ("my error", st9.Msg());

  Status st10(std::move(st9));
  ASSERT_EQ(Status::kIndexOverflow, st10.Code());
  ASSERT_EQ("my error", st10.Msg());
}

TEST(StatusTest, EqualAndNotEqual) {
  Status st1 = Status::Ok();
  Status st2 = Status::Ok();
  Status st3 = Status::IndexOverflow("my error");
  Status st4 = Status::IndexOverflow("my error");
  ASSERT_TRUE(st1 == st2);
  ASSERT_FALSE(st1 != st2);
  ASSERT_TRUE(st3 == st4);
  ASSERT_FALSE(st3 != st4);
  ASSERT_TRUE(st1 != st3);
  ASSERT_FALSE(st1 == st3);
  ASSERT_TRUE(st3 != st1);
  ASSERT_FALSE(st3 == st1);
}

TEST(StatusTest, ToString) {
  Status st1 = Status::Ok();
  Status st2 = Status::IndexOverflow("my error");
  ASSERT_EQ("OK", st1.ToString());
  ASSERT_EQ("ErrorCode [2]: my error", st2.ToString());
}

TEST(StatusTest, IsOk) {
  Status st1 = Status::Ok();
  Status st2 = Status::IndexOverflow("my error");
  ASSERT_TRUE(st1.IsOk());
  ASSERT_FALSE(st2.IsOk());
}

TEST(StatusTest, StatusOk) {
  Status st = Status::Ok();
  ASSERT_EQ(Status::kOk, st.Code());
  ASSERT_EQ("", st.Msg());
}

TEST(StatusTest, StatusArgumentError) {
  Status st = Status::ArgumentError("my error");
  ASSERT_EQ(Status::kArgumentError, st.Code());
  ASSERT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusIndexOverflow) {
  Status st = Status::IndexOverflow("my error");
  ASSERT_EQ(Status::kIndexOverflow, st.Code());
  ASSERT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusInternal) {
  Status st = Status::Internal("my error");
  ASSERT_EQ(Status::kInternal, st.Code());
  ASSERT_EQ("my error", st.Msg());
}


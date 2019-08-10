#include "gtest/gtest.h"
#include "ps-plus/common/status.h"

using ps::Status;

TEST(StatusTest, DefaultConstructor) {
  Status st;
  EXPECT_EQ(Status::kOk, st.Code());
  EXPECT_EQ("", st.Msg());
}

TEST(StatusTest, CopyAndMove) { 
  Status st1 = Status::Ok();
  EXPECT_EQ(Status::kOk, st1.Code());
  EXPECT_EQ("", st1.Msg());

  Status st2 = Status::IndexOverflow("my error");
  EXPECT_EQ(Status::kIndexOverflow, st2.Code());
  EXPECT_EQ("my error", st2.Msg());

  Status st3;
  st3 = st1;
  EXPECT_EQ(Status::kOk, st3.Code());
  EXPECT_EQ("", st3.Msg());

  st3 = st2;
  EXPECT_EQ(Status::kIndexOverflow, st3.Code());
  EXPECT_EQ("my error", st3.Msg());

  Status st4;
  st4 = std::move(st1);
  EXPECT_EQ(Status::kOk, st4.Code());
  EXPECT_EQ("", st4.Msg());

  Status st5;
  st5 = std::move(st2);
  EXPECT_EQ(Status::kIndexOverflow, st5.Code());
  EXPECT_EQ("my error", st5.Msg());

  Status st6;
  st6 = std::move(st5);
  EXPECT_EQ(Status::kIndexOverflow, st6.Code());
  EXPECT_EQ("my error", st6.Msg());

  Status st7(st4);
  EXPECT_EQ(Status::kOk, st7.Code());
  EXPECT_EQ("", st7.Msg());

  Status st8(std::move(st7));
  EXPECT_EQ(Status::kOk, st8.Code());
  EXPECT_EQ("", st8.Msg());

  Status st9(st6);
  EXPECT_EQ(Status::kIndexOverflow, st9.Code());
  EXPECT_EQ("my error", st9.Msg());

  Status st10(std::move(st9));
  EXPECT_EQ(Status::kIndexOverflow, st10.Code());
  EXPECT_EQ("my error", st10.Msg());
}

TEST(StatusTest, EqualAndNotEqual) { 
  Status st1 = Status::Ok();
  Status st2 = Status::Ok();
  Status st3 = Status::IndexOverflow("my error");
  Status st4 = Status::IndexOverflow("my error");
  EXPECT_TRUE(st1 == st2);
  EXPECT_FALSE(st1 != st2);
  EXPECT_TRUE(st3 == st4);
  EXPECT_FALSE(st3 != st4);
  EXPECT_TRUE(st1 != st3);
  EXPECT_FALSE(st1 == st3);
  EXPECT_TRUE(st3 != st1);
  EXPECT_FALSE(st3 == st1);
}

TEST(StatusTest, ToString) { 
  Status st1 = Status::Ok();
  Status st2 = Status::IndexOverflow("my error");
  EXPECT_EQ("OK", st1.ToString());
  EXPECT_EQ("ErrorCode [2]: my error", st2.ToString());
}

TEST(StatusTest, IsOk) { 
  Status st1 = Status::Ok();
  Status st2 = Status::IndexOverflow("my error");
  EXPECT_TRUE(st1.IsOk());
  EXPECT_FALSE(st2.IsOk());
}

TEST(StatusTest, StatusOk) { 
  Status st = Status::Ok();
  EXPECT_EQ(Status::kOk, st.Code());
  EXPECT_EQ("", st.Msg());
}

TEST(StatusTest, StatusArgumentError) { 
  Status st = Status::ArgumentError("my error");
  EXPECT_EQ(Status::kArgumentError, st.Code());
  EXPECT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusIndexOverflow) { 
  Status st = Status::IndexOverflow("my error");
  EXPECT_EQ(Status::kIndexOverflow, st.Code());
  EXPECT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusNotFound) { 
  Status st = Status::NotFound("my error");
  EXPECT_EQ(Status::kNotFound, st.Code());
  EXPECT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusDataLoss) { 
  Status st = Status::DataLoss("my error");
  EXPECT_EQ(Status::kDataLoss, st.Code());
  EXPECT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusAlreadyExist) { 
  Status st = Status::AlreadyExist("my error");
  EXPECT_EQ(Status::kAlreadyExist, st.Code());
  EXPECT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusNotImplemented) { 
  Status st = Status::NotImplemented("my error");
  EXPECT_EQ(Status::kNotImplemented, st.Code());
  EXPECT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusUdfNotRegistered) { 
  Status st = Status::UdfNotRegistered("my error");
  EXPECT_EQ(Status::kUdfNotRegistered, st.Code());
  EXPECT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusNetworkError) { 
  Status st = Status::NetworkError("my error");
  EXPECT_EQ(Status::kNetworkError, st.Code());
  EXPECT_EQ("my error", st.Msg());
}

TEST(StatusTest, StatusVersionMismatch) { 
  Status st = Status::VersionMismatch("my error");
  EXPECT_EQ(Status::kVersionMismatch, st.Code());
  EXPECT_EQ("my error", st.Msg());
}


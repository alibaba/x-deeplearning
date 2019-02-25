/*
 * \file broadcast_test.cc
 * \brief The broadcast test unit
 */
#include "gtest/gtest.h"

#include "blaze/math/broadcast.h"

namespace blaze {

TEST(TestMBroadcasting, BroadcastShape) {
  std::vector<TIndex> a = { 1UL, 2UL, 3UL };
  std::vector<TIndex> b = { 3UL, 1UL, 3UL };
  std::vector<TIndex> c;
  EXPECT_TRUE(MBroadcasting::BroadcastShape(a, b, c));

  EXPECT_EQ(3, c.size());
  EXPECT_EQ(3, c[0]);
  EXPECT_EQ(2, c[1]);
  EXPECT_EQ(3, c[2]);

  a[0] = 2UL;
  EXPECT_FALSE(MBroadcasting::BroadcastShape(a, b, c));
}

TEST(TestUBroadcasting, BroadcastShape) {
  std::vector<TIndex> a = { 1UL, 3UL };
  std::vector<TIndex> b = { 1UL, 2UL, 3UL };
  std::vector<TIndex> c;
  EXPECT_FALSE(UBroadcasting::BroadcastShape(a, b, c));
  // TODO: Add more
}

TEST(TestUBroadcasting, DimEqual) {
  std::vector<TIndex> a = { 2UL, 3UL };
  std::vector<TIndex> b = { 1UL, 2UL, 3UL };
  EXPECT_TRUE(UBroadcasting::DimEqual(a, b));
}



}  // namespace blaze

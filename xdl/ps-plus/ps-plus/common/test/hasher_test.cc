#include "gtest/gtest.h"
#include "ps-plus/common/hasher.h"

using ps::Hasher;

TEST(HasherTest, Hasher) {
  int value = Hasher::Hash128(123456789, 987654321);
  ASSERT_NE(value, 0);
  ASSERT_NE(value, 1);

  value = Hasher::Hash64(123456789);
  ASSERT_EQ(value, 52501);
  value = Hasher::Hash64(-123456789);
  ASSERT_EQ(value, 13035);
  value = Hasher::Hash64(0);
  ASSERT_EQ(value, 0);
  value = Hasher::Hash64(65536);
  ASSERT_EQ(value, 0);
}

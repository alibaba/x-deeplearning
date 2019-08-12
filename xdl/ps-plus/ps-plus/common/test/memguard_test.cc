#include "gtest/gtest.h"
#include "ps-plus/common/memguard.h"

using ps::serializer::MemGuard;

TEST(MemGuardTest, MemGuard) {
  MemGuard mg1;
  MemGuard mg2(mg1);

  MemGuard mg3;
  mg3 = mg1;
}

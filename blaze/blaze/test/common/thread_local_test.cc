/*
 * \file thread_local_test.cc
 * \brief The thread_local test module
 */
#include "gtest/gtest.h"

#include <math.h>

#include "blaze/common/thread_local.h"

namespace blaze {

TEST(TestThreadLocal, all) {
  int* p = ThreadLocalStore<int>::Get();
  *p = 1;
  EXPECT_EQ(1, *ThreadLocalStore<int>::Get());
}

}  // namespace blaze

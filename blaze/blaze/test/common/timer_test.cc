/*
 * \file timer_test.cc
 * \brief The timer test module
 */
#include "gtest/gtest.h"

#include <math.h>

#include "blaze/common/timer.h"

namespace blaze {

TEST(TestTimer, TestAll) {
  Timer timer;
  timer.Start();
  timer.Stop();
  double dur = timer.GetElapsedTime();
  EXPECT_TRUE(dur <= 1);

  timer.Reset();
  EXPECT_EQ(0, timer.GetTotalTime());

  timer.ReStart();
  timer.Stop();
  dur = timer.GetElapsedTime();
  EXPECT_TRUE(dur <= 1);
}

}  // namespace blaze


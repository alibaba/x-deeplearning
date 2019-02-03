/*
 * \file math_test.cc
 * \brief The math test unit
 */
#include "gtest/gtest.h"

#include "blaze/common/log.h"

namespace blaze {

TEST(TestAll, Test) {
#if __CUDA_ARCH__ >= 200
  LOG_INFO("Hello World");
#endif
}

}


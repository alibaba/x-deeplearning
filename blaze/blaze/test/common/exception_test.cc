/*
 * \file exception_test.cc
 * \brief The exception test module
 */
#include "gtest/gtest.h"

#include <math.h>

#include "blaze/common/exception.h"
#include "blaze/common/log.h"

namespace blaze {

TEST(TestException, All) {
  try {
    BLAZE_THROW("hello world");
  } catch (Exception& e) {
    const char* msg = e.what();
    LOG_INFO("msg=%s", msg);
  }
}

}  // namespace blaze



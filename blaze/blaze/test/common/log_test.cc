/*
 * \file log_test.cc
 * \brief The log test module
 */
#include "gtest/gtest.h"

#include "blaze/common/log.h"

namespace blaze {

TEST(TestLog, All) {
  Logger logger;
  logger.ResetLogFile("./log.txt");
  logger.Write(kInfo, "Hello World");
  logger.Debug("Hello World Debug");
  logger.Info("Hello World Info");
  logger.Error("Hello World Error");
}

}  // namespace blaze

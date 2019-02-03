/*
 * \file thread_pool_test.cc
 * \brief The thread pool test module
 */
#include "gtest/gtest.h"

#include "blaze/common/thread_pool.h"
#include "blaze/common/log.h"

namespace blaze {

TEST(TestThreadPool, All) {
  ThreadExecutor thread_executor;
  thread_executor.commit([]() {
    LOG_INFO("hello world");                    
  });
  thread_executor.shutdown();
}

}  // namespace blaze


#include "gtest/gtest.h"
#include "ps-plus/common/thread_pool.h"

using ps::ThreadPool;

TEST(ThreadPoolTest, ThreadPool) {
  const size_t size = 3ul << 30; /* 3G */
  char *dest = (char*)malloc(size); /* 1 MegaBytes */
  char *src = (char*)malloc(size);
  ASSERT_NE(dest, nullptr);
  ASSERT_NE(src, nullptr);
  memset(src, 'c', size);

  ps::QuickMemcpy(dest, src, size);

  ASSERT_EQ(dest[0], 'c');
  ASSERT_EQ(dest[size - 1], 'c');

  free(dest);
  free(src);
}

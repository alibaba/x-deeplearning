/*
 * \file allocator_test.cc
 * \brief The allocator test module
 */
#include "gtest/gtest.h"

#include "blaze/common/allocator.h"
#include "blaze/common/timer.h"
#include "blaze/common/log.h"

namespace blaze {

TEST(TestCPUAllocator, All) {
  void* space = Allocator<kCPU>::Alloc(10, 0);
  EXPECT_TRUE(space != nullptr);
  Allocator<kCPU>::Free(space, 10, 0);
}

TEST(TestCPUSlabAllocator, All) {
  void* space = SlabAllocator<kCPU, 0>::Get()->Alloc(10);
  EXPECT_TRUE(space != nullptr);
  SlabAllocator<kCPU, 0>::Get()->Free(space, 10);
  void* space2 = SlabAllocator<kCPU, 0>::Get()->Alloc(10);
  EXPECT_EQ(space, space2);

  auto space3 = SlabAllocator<kCPU, 0>::Get()->Alloc(10);
  auto space4 = SlabAllocator<kCPU, 0>::Get()->Alloc(10);
  EXPECT_TRUE(static_cast<char*>(space3) + 16 == static_cast<char*>(space4));

  space3 = SlabAllocator<kCPU, 0>::Get()->Alloc(16);
  space4 = SlabAllocator<kCPU, 0>::Get()->Alloc(10);
  EXPECT_TRUE(static_cast<char*>(space3) + 16 == static_cast<char*>(space4));
}

#ifdef USE_CUDA
TEST(TestCUDAAllocator, All) {
  void* space = Allocator<kCUDA>::Alloc(10, 0);
  EXPECT_TRUE(space != nullptr);
  Allocator<kCUDA>::Free(space, 10, 0);
}

TEST(TestCUDASlabAllocator, All) {
  return;
  void* space = SlabAllocator<kCUDA, 0>::Get()->Alloc(10);
  EXPECT_TRUE(space != nullptr);
  SlabAllocator<kCUDA, 0>::Get()->Free(space, 10);

  Timer timer;
  timer.Start();
  int kLoop = 100000;
  for (int i = 0; i < kLoop; ++i) {
    auto size = (i + 1) % 1024;
    auto space = SlabAllocator<kCUDA, 0>::Get()->Alloc(size);
    SlabAllocator<kCUDA, 0>::Get()->Free(space, size);
  }
  timer.Stop();
  LOG_INFO("QPS: %f %f", (kLoop) / timer.GetElapsedTime(), timer.GetElapsedTime());
}
#endif

}  // namespace blaze

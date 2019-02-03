/*
 * \file queue_test.cc
 * \brief The queue test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/common/queue.h"
#include "blaze/common/log.h"

namespace blaze {

TEST(Queue, Push) {
  Queue<int> queue;
  int a = 1;
  queue.Push(a);
  EXPECT_EQ(queue.Size(), 1);
  a = 2;
  queue.Push(a);
  EXPECT_EQ(queue.Size(), 2);

  bool success = queue.Pop(a);
  EXPECT_TRUE(success);
  EXPECT_EQ(a, 1);

  success = queue.Front(a);
  EXPECT_TRUE(success);
  EXPECT_EQ(a, 2);

  EXPECT_FALSE(queue.Empty());

  success = queue.Pop(a);
  EXPECT_TRUE(success);
  EXPECT_EQ(a, 2);

  success = queue.TryPop(a);
  EXPECT_FALSE(success);

  EXPECT_TRUE(queue.Empty());
}

void JobRunner(Queue<int>* queue) {
  bool alive = queue->Alive();
  while (alive) {
    int a;
    bool success = queue->Pop(a);
    alive = queue->Alive();
    LOG_INFO("success %d %d", a, alive);
  }
  int a;
  while (queue->TryPop(a)) {
    LOG_INFO("success %d", a);
  }
  LOG_INFO("exiting");
}

TEST(Queue, ExitAlive) {
  Queue<int> queue;
  std::thread thread1(JobRunner, &queue);
  std::thread thread2(JobRunner, &queue);
  int a = 2;
  queue.Push(a);
  a = 3;
  queue.Push(a);
  queue.Exit();
  // exit thread1/thread2
  thread1.join();
  thread2.join();
}

}  // namespace blaze

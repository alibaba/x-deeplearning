// Test unit for SharedBatchScheduler
//
#include "blaze/batching/shared_batch_scheduler.h"

#include "gtest/gtest.h"

namespace blaze {
namespace batching {

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) { }
  ~FakeTask() override = default;
  size_t size() const override { return size_; }

 private:
  const size_t size_;
};

Status ScheduleTask(size_t task_size, BatchScheduler<FakeTask>* scheduler) {
  std::unique_ptr<FakeTask> task(new FakeTask(task_size));
  Status status = scheduler->Schedule(&task);
  EXPECT_EQ(status == kOk, task == nullptr);
  return status;
}

TEST(SharedBatchScheduler, Test) {
  bool callback_called = false;
  auto callback = [&callback_called](std::unique_ptr<Batch<FakeTask>> batch) {
    LOG_INFO("callback called");
    callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    ASSERT_EQ(3, batch->task(0).size());
    ASSERT_EQ(5, batch->task(1).size());
  };
  {
    SharedBatchScheduler<FakeTask>::Options options;
    LOG_INFO("thread_num = %d", options.num_batch_threads);
    std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
    EXPECT_TRUE(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));

    SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    queue_options.max_batch_size = 10;
    queue_options.batch_timeout_micros = 100 * 1000; /// 100 milliseconds
    queue_options.max_enqueued_batches = 3;

    /// queue1
    std::unique_ptr<BatchScheduler<FakeTask>> queue1;
    EXPECT_TRUE(scheduler->AddQueue(queue_options, callback, &queue1));

    EXPECT_EQ(0, queue1->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10, queue1->SchedulingCapacity());
    EXPECT_EQ(kOk, ScheduleTask(3, queue1.get()));
    EXPECT_EQ(1, queue1->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10 - 3, queue1->SchedulingCapacity());
    EXPECT_EQ(kOk, ScheduleTask(5, queue1.get()));
    EXPECT_EQ(2, queue1->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10 - 3 - 5, queue1->SchedulingCapacity());
  
    /// queue2
    std::unique_ptr<BatchScheduler<FakeTask>> queue2;
    EXPECT_TRUE(scheduler->AddQueue(queue_options, callback, &queue2));

    EXPECT_EQ(0, queue2->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10, queue2->SchedulingCapacity());
    EXPECT_EQ(kOk, ScheduleTask(3, queue2.get()));
    EXPECT_EQ(1, queue2->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10 - 3, queue2->SchedulingCapacity());
    EXPECT_EQ(kOk, ScheduleTask(5, queue2.get()));
    EXPECT_EQ(2, queue2->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10 - 3 - 5, queue2->SchedulingCapacity());
  }
  EXPECT_TRUE(callback_called);
}

}  // namespace batching
}  // namespace blaze

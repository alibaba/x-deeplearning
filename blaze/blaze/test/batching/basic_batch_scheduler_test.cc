// Test unit for BasicBatchScheduler
//
#include "blaze/batching/basic_batch_scheduler.h"

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
  int buf[8192];
};

// Creates a FaskTask of size 'TaskSize', and calls scheduler->Schedule() on
// that task. Returns the resulting status.
Status ScheduleTask(size_t task_size, BatchScheduler<FakeTask>* scheduler) {
  std::unique_ptr<FakeTask> task(new FakeTask(task_size));
  Status status = scheduler->Schedule(&task);
  EXPECT_EQ(task == nullptr, status == kOk);
  return status;
}

TEST(BasicBatchSchedulerTest, Basic) {
  bool callback_called = false;
  auto callback = [&callback_called](std::unique_ptr<Batch<FakeTask>> batch) {
    callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(3, batch->task(0).size());
    EXPECT_EQ(5, batch->task(1).size());
  };
  {
    BasicBatchScheduler<FakeTask>::Options options;
    options.max_batch_size = 10;
    options.batch_timeout_micros = 100 * 1000; /// 100 milliseconds
    options.num_batch_threads = 1;
    options.max_enqueued_batches = 3;
    std::unique_ptr<BasicBatchScheduler<FakeTask>> scheduler;
    EXPECT_TRUE(BasicBatchScheduler<FakeTask>::Create(options, callback, &scheduler));
    EXPECT_EQ(0, scheduler->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10, scheduler->SchedulingCapacity());
    EXPECT_EQ(kOk, ScheduleTask(3, scheduler.get()));
    EXPECT_EQ(1, scheduler->NumEnqueuedTasks());
    EXPECT_EQ(3* 10 - 3, scheduler->SchedulingCapacity());
    EXPECT_EQ(kOk, ScheduleTask(5, scheduler.get()));
    EXPECT_EQ(2, scheduler->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10 - 3 - 5, scheduler->SchedulingCapacity());
  }
  EXPECT_TRUE(callback_called);
}

}  // namespace batching
}  // namespace blaze

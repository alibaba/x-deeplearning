/*
 * \file cpu_event_test.cc
 * \brief The cpu event test module
 */
#include "gtest/gtest.h"

#include "blaze/common/context.h"
#include "blaze/common/cpu_event.h"

namespace blaze {

TEST(TestCPUEvent, All) {
  DeviceOption device_option;
  device_option.set_device_type(kCPU);

  Event event(device_option);
  CPUContext context;

  context.Record(&event);
  event.SetFinished();

  context.WaitEvent(event);
  event.Finish();

  event.Reset();
  event.Record(kCPU, &context);
  event.SetFinished();
  event.Wait(kCPU, &context);
}

}  // namespace blaze


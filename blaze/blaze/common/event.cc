/*!
 * \file event.cc
 * \brief The event.
 */
#include "blaze/common/event.h"

namespace blaze {

EventCreateFunction Event::event_creator_[kMaxDeviceTypes];
EventRecordFunction Event::event_recorder_[kMaxDeviceTypes];
EventWaitFunction Event::event_waiter_[kMaxDeviceTypes][kMaxDeviceTypes];
EventFinishFunction Event::event_finisher_[kMaxDeviceTypes];
EventQueryFunction Event::event_querier_[kMaxDeviceTypes];
EventErrorMessageFunction Event::event_err_msg_getter_[kMaxDeviceTypes];
EventSetFinishedFunction Event::event_finished_setter_[kMaxDeviceTypes];
EventResetFunction Event::event_resetter_[kMaxDeviceTypes];

}  // namespace blaze


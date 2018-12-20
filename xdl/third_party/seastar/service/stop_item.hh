#ifndef ML_PS5_NET_STOP_ITEM_H
#define ML_PS5_NET_STOP_ITEM_H

#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "core/ps_coding/unittest/echo_request_serializer.hh"
#include "service/session_context.hh"

#include <iostream>

namespace ps
{
namespace network
{

class StopItem : public ps::network::SeastarWorkItem
{
public:
    StopItem(int64_t userThreadId) 
    : ps::network::SeastarWorkItem(NULL, -1)
    , mUserThreadId(userThreadId)
    {}

    seastar::future<> Run() override
    {
        seastar::smp::stop_user_thread_id = mUserThreadId;
        seastar::smp::stop_sequence = GetSequence();
        seastar::engine().exit(0);
        return seastar::make_ready_future<>();
    }

private:
    int64_t mUserThreadId;
};

} // namespace net
} // ps

#endif // ML_PS5_NET_STOP_ITEM_H


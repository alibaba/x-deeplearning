#ifndef PS_ACTION_ACTION_PULL_REQUEST_ITEM_H
#define PS_ACTION_ACTION_PULL_REQUEST_ITEM_H

#include <iostream>
#include "core/ps_queue_hub/seastar_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "service/client_network_context.hh"
#include "action/get_request_serializer.hh"
#include "action/utility.hh"

namespace ps
{
namespace action
{
using namespace std;

template<typename T>
class PullRequestItem : public ps::network::SeastarWorkItem
{
public:
    PullRequestItem(ps::network::ClientNetworkContext& cnc, int32_t user,
            const vector<T>& v, int64_t begin, int64_t end,
            int64_t serverId, int32_t shardKey = 0)
        : ps::network::SeastarWorkItem(&cnc, serverId),
        mWorkerNetworkContext(cnc), kUserId(user),
        kSource(v), kBegin(begin), kEnd(end), mShardKey(shardKey)
    {}

    seastar::future<> Run() override
    {
        //utility::CoutCurrentTimeOnNanoSec("__________echo_request_item_run_start_time", false);
        ps::action::GetRequestSerializer<T>* serializer
            = new ps::action::GetRequestSerializer<T>(kSource, kBegin, kEnd, mShardKey);
        serializer->SetUserThreadId(kUserId);
        serializer->SetSequence(this->GetSequence());

        serializer->GetTimeStamp()->mWorkerEnqueueTime = this->mEnqueueTime;
        serializer->GetTimeStamp()->mWorkerDequeueTime = this->mDequeueTime;
        serializer->GetTimeStamp()->mWorkerSessionWriteTime
            = utility::CoutCurrentTimeOnNanoSec("_____echo_request_item_run_before_session_write_time", false);

        return this->GetSessionContext()->Write(serializer);
    }

    void Complete() {}

private:
    ps::network::ClientNetworkContext& mWorkerNetworkContext;
    const int32_t kUserId;
    const vector<T>& kSource;
    const int64_t kBegin;
    const int64_t kEnd;
    int32_t mShardKey;
};

} // namespace action
} // ps

#endif // PS_ACTION_ACTION_PULL_REQUEST_ITEM_H

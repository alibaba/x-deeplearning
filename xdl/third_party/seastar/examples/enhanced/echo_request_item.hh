#ifndef ML_PS5_CODING_UNITTEST_ECHO_REQUEST_ITEM_H
#define ML_PS5_CODING_UNITTEST_ECHO_REQUEST_ITEM_H

#include <iostream>
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "examples/dense/echo_request_serializer.hh"
#include "service/network_context.hh"

extern uint64_t CoutCurrentTimeOnNanoSec(const char * prompt, bool output);

namespace ps
{
namespace coding
{
namespace unittest
{
using namespace std;

template<typename T>
class EchoRequestItem : public ps::network::SeastarWorkItem
{
public:
    EchoRequestItem(int32_t instanceId,  ps::network::NetworkContext* networkContext, int64_t serverId,
            const vector<T>& v, int64_t begin, int64_t end, int32_t shardId = 0)
        : ps::network::SeastarWorkItem(networkContext, serverId), mInstanceId(instanceId),
        mSource(v), mBegin(begin), mEnd(end), mShardId(shardId)
    {
        mReceiverId = 0;
    }

    seastar::future<> Run() override
    {
        CoutCurrentTimeOnNanoSec("__________echo_request_item_run_start_time", false);

        ps::coding::unittest::EchoRequestSerializer<T>* serializer
            = new ps::coding::unittest::EchoRequestSerializer<T>(mSource, mBegin, mEnd, mShardId);
        serializer->SetUserThreadId(this->GetInstanceId());
        serializer->SetSequence(this->GetSequence());

        serializer->GetTimeStamp()->mWorkerEnqueueTime = this->mEnqueueTime;
        serializer->GetTimeStamp()->mWorkerDequeueTime = this->mDequeueTime;
        serializer->GetTimeStamp()->mWorkerSessionWriteTime
            = CoutCurrentTimeOnNanoSec("__________echo_request_item_run_before_session_write_time", false);

        return GetSessionContext()->Write(serializer);
    }

    void Complete() {}

    int GetReceiverId() const { return mReceiverId; }
    void SetReceiverId(int receiverId) { mReceiverId = receiverId; }
    int32_t GetInstanceId() const { return mInstanceId; }
private:
    int mReceiverId;
    int32_t mInstanceId;
    const vector<T>& mSource;
    int64_t mBegin;
    int64_t mEnd;
    int32_t mShardId;
};

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_REQUEST_ITEM_H

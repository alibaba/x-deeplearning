#ifndef PS_ACTION_ACTION_GET_PROCESS_ITEM_H
#define PS_ACTION_ACTION_GET_PROCESS_ITEM_H

#include <iostream>
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "service/session_context.hh"
#include "action/utility.hh"
#include "action/update_request_serializer.hh"

namespace ps
{
namespace action
{
using namespace std;

template<typename T>
class GetProcessItem : public ps::network::PsWorkItem
{
public:
    GetProcessItem(int32_t instanceId, ps::network::SessionContext* sc,
            ps::action::UpdateRequestSerializer<T>* serializer) :
        ps::network::PsWorkItem(), mReceiverId(0), mInstanceId(instanceId), mSessionContext(sc),
        mSerializer(serializer)
    {}

    void Run() override
    {
        // TODO: access storage
        mSerializer->GetTimeStamp()->mServerEnqueueTime = this->mEnqueueTime;
        mSerializer->GetTimeStamp()->mServerDequeueTime = this->mDequeueTime;
        T* sourceData = const_cast<T*>(mSerializer->GetSourceData());
        const int32_t sourceCount = mSerializer->GetSourceCount();
        for (int32_t i = 0; i < sourceCount; ++i)
        {
            *(sourceData + i) *= 2;
        }
        return;
    }
    seastar::future<> Complete() override
    {
        mSerializer->GetTimeStamp()->mServerEnqueueTime2 = this->mEnqueueTime;
        mSerializer->GetTimeStamp()->mServerDequeueTime2 = this->mDequeueTime;
        const T* sourceData = mSerializer->GetSourceData();
        const int32_t sourceCount = mSerializer->GetSourceCount();
        mSerializer->SetTestMessage(reinterpret_cast<const char*>(sourceData), sourceCount * sizeof(T));
        mSerializer->GetTimeStamp()->mServerSessionWriteTime
            = utility::CoutCurrentTimeOnNanoSec("_______server_before_session_write", false);
        return mSessionContext->Write(mSerializer);
    }

    int GetReceiverId() const { return mReceiverId; }
    void SetReceiverId(int receiverId) { mReceiverId = receiverId; }
    int32_t GetInstanceId() const { return mInstanceId; }
private:
    int mReceiverId;
    int32_t mInstanceId;
    ps::network::SessionContext* mSessionContext;
    ps::action::UpdateRequestSerializer<T>* mSerializer;
    uint64_t mSideSequ;
    int32_t mUserThreadId;
};

} // namespace action
} // ps

#endif // PS_ACTION_ACTION_GET_PROCESS_ITEM_H

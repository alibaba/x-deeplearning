#ifndef ML_PS5_CODING_UNITTEST_ECHO_REQUEST_ITEM_H
#define ML_PS5_CODING_UNITTEST_ECHO_REQUEST_ITEM_H

#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "core/ps_coding/unittest/echo_request_serializer.hh"
#include "service/network_context.hh"

namespace ps
{
namespace coding
{
namespace unittest
{

class EchoRequestItem : public ps::network::SeastarWorkItem
{
public:
    EchoRequestItem(int32_t instanceId,  ps::network::NetworkContext* networkContext, int64_t serverId) : 
    ps::network::SeastarWorkItem(networkContext, serverId), mInstanceId(instanceId)
    {
        mReceiverId = 0;
    }

    seastar::future<> Run() override
    {
        static const char kTestData[128] =
            "0123456789" "abcdefghij" "0123456789" "abcdefghij"
            "abcdefghij" "0123456789" "abcdefghij" "0123456789"
            "0123456789" "abcdefghij" "0123456789" "abcdefghij"
            "abcdefg" ;
        ps::coding::unittest::EchoRequestSerializer* serializer
            = new ps::coding::unittest::EchoRequestSerializer;
        serializer->SetUserThreadId(this->GetInstanceId());
        serializer->SetTestMessage(kTestData);

        // set sequence id: need to be refined
        // user should not set the value!
        serializer->SetSequence(this->GetSequence());

        return GetSessionContext()->Write(serializer).then([serializer] () 
        {
           // DONT delete here, it will be delete in packet::deleter() 
           // delete serializer;
           return seastar::make_ready_future<>();
        });
    }

    int GetReceiverId() const { return mReceiverId; }
    void SetReceiverId(int receiverId) { mReceiverId = receiverId; }
    int32_t GetInstanceId() const { return mInstanceId; }
private:
    int mReceiverId;
    int32_t mInstanceId;
};

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_REQUEST_ITEM_H

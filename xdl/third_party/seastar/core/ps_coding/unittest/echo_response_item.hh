#ifndef ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_ITEM_H
#define ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_ITEM_H

#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "core/ps_coding/unittest/echo_request_serializer.hh"
#include "service/session_context.hh"

using namespace std;

namespace ps
{
namespace coding
{
namespace unittest
{

class EchoResponseItem : public ps::network::PsWorkItem
{
public:
    EchoResponseItem(int32_t instanceId, ps::network::SessionContext* sc,
            ps::coding::unittest::EchoResponseSerializer* serializer) :
            //uint64_t a, int32_t b) :
        ps::network::PsWorkItem(), mReceiverId(0), mInstanceId(instanceId), mSessionContext(sc),
        mSerializer(serializer)//, mSideSequ(a), mUserThreadId(b)
    {}

    void Run() override
    {
        std::cout << "EchoResponseItem Run, do nothing." << std::endl;
        return;
    }
    seastar::future<> Complete()
    {
        /*
        static const char kTestData[128] =
            "0123456789" "abcdefghij" "0123456789" "abcdefghij"
            "abcdefghij" "0123456789" "abcdefghij" "0123456789"
            "0123456789" "abcdefghij" "0123456789" "abcdefghij"
            "abcdefg" ;
        ps::coding::unittest::EchoResponseSerializer* serializer
            = new ps::coding::unittest::EchoResponseSerializer;
        serializer->SetTestMessage(kTestData);
        serializer->SetSequence(mSideSequ);
        serializer->SetUserThreadId(mUserThreadId);
        */

        std::cout << "EchoResponseItem Complete"
            << " mSessionContext:" << mSessionContext
            << " mSerializer:" << mSerializer
            //<< " mSideSequ:" << mSideSequ
            //<< " mUserThreadId:" << mUserThreadId
            << std::endl;
        //mSessionContext->Write(serializer).then([serializer] () {
                //delete serializer;
        //mSessionContext->Write(mSerializer).then([this] () {
                //cout << "Server Delete" << endl;
                //delete this->mSerializer;
        //});

        return mSessionContext->Write(mSerializer);
    }

    int GetReceiverId() const { return mReceiverId; }
    void SetReceiverId(int receiverId) { mReceiverId = receiverId; }
    int32_t GetInstanceId() const { return mInstanceId; }
private:
    int mReceiverId;
    int32_t mInstanceId;
    ps::network::SessionContext* mSessionContext;
    ps::coding::unittest::EchoResponseSerializer* mSerializer;
    uint64_t mSideSequ;
    int32_t mUserThreadId;
};

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_ITEM_H

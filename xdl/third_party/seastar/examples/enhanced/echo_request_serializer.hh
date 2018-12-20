#ifndef ML_PS5_CODING_UNITTEST_ECHO_REQUEST_SERIALIZER_H
#define ML_PS5_CODING_UNITTEST_ECHO_REQUEST_SERIALIZER_H

#include <iostream>
#include "core/ps_coding/message_serializer.hh"
#include "examples/dense/echo_request_processor.hh"

using namespace std;

namespace ps
{
namespace coding
{
namespace unittest
{

template<typename T>
class EchoRequestSerializer : public MessageSerializer
{
public:
    EchoRequestSerializer(const vector<T>& v, int64_t begin, int64_t end, int32_t shardId = 0) :
        mSource(v), mBegin(begin), mEnd(end),
        mTestMessage(reinterpret_cast<const char*>(mSource.data()) + sizeof(T) * mBegin),
        mTestMessageLength(sizeof(T) * (mEnd - mBegin)),
        mShardId(shardId)
    {}
    void Serialize() override
    {
        SetProcessorClassId(kEchoRequestProcessor_ClassId);
        AppendMeta(reinterpret_cast<const void *>(&mShardId), sizeof(mShardId));
        AppendFragment(mTestMessage, mTestMessageLength);
    }

private:
    const vector<T>& mSource;
    int64_t mBegin;
    int64_t mEnd;
    const char* mTestMessage = nullptr;
    int32_t mTestMessageLength = 128;
    int32_t mShardId;
};

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_REQUEST_SERIALIZER_H

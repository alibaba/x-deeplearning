#ifndef PS_ACTION_ACTION_GET_REQUEST_SERIALIZER_H
#define PS_ACTION_ACTION_GET_REQUEST_SERIALIZER_H

#include <iostream>
#include "core/ps_coding/message_serializer.hh"
#include "action/request_processor.hh"

namespace ps
{
namespace action
{
using namespace std;

template<typename T>
class GetRequestSerializer : public ps::coding::MessageSerializer
{
public:
    GetRequestSerializer(const vector<T>& v, int64_t begin, int64_t end, int32_t shardKey = 0) :
        kSourceVec(v), kBegin(begin), kEnd(end),
        mTestMessage(reinterpret_cast<const char*>(kSourceVec.data()) + sizeof(T) * kBegin),
        mTestMessageLength(sizeof(T) * (kEnd - kBegin)),
        mShardKey(shardKey)
    {}
    void Serialize() override
    {
        SetProcessorClassId(kActionRequestProcessor_ClassId);
        //AppendMeta(reinterpret_cast<const void *>(&mShardKey), sizeof(mShardKey));
        //AppendFragment(mTestMessage, mTestMessageLength);
    }

private:
    const vector<T>& kSourceVec;
    const int64_t kBegin;
    const int64_t kEnd;
    const char* mTestMessage = nullptr;
    int64_t mTestMessageLength = 0;
    int32_t mShardKey;
};

} // namespace
} // namespace

#endif // PS_ACTION_ACTION_GET_REQUEST_SERIALIZER_H

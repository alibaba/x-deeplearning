#ifndef PS_ACTION_ACTION_PUSH_REQUEST_SERIALIZER_H
#define PS_ACTION_ACTION_PUSH_REQUEST_SERIALIZER_H

#include <iostream>
#include "core/ps_coding/message_serializer.hh"
#include "action/action_meta.hh"
#include "action/request_processor.hh"

namespace ps
{
namespace action
{
using namespace std;

template<typename T>
class UpdateRequestSerializer : public ps::coding::MessageSerializer
{
public:
    UpdateRequestSerializer(const vector<T>& v, int64_t begin, int64_t end, int32_t shardKey = 0) :
        mMeta(ActionMetaHelper::Get<T>(kPushWithRes)),
        kSourceVec(v), kBegin(begin), kEnd(end),
        mTestMessage(reinterpret_cast<const char*>(kSourceVec.data()) + sizeof(T) * kBegin),
        mTestMessageLength(sizeof(T) * (kEnd - kBegin)),
        mShardKey(shardKey)
    {}
    void Serialize() override
    {
        SetProcessorClassId(kActionRequestProcessor_ClassId);
        AppendMeta(reinterpret_cast<const void *>(&mMeta), sizeof(mMeta));
        AppendFragment(mTestMessage, mTestMessageLength);
    }

private:
    ActionMeta mMeta;
    const vector<T>& kSourceVec;
    const int64_t kBegin;
    const int64_t kEnd;
    const char* mTestMessage = nullptr;
    int64_t mTestMessageLength = 0;
    int32_t mShardKey;
};

} // namespace action
} // ps

#endif // PS_ACTION_ACTION_PUSH_REQUEST_SERIALIZER_H

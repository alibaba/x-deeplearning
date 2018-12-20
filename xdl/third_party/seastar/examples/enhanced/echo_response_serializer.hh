#ifndef ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_SERIALIZER_H
#define ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_SERIALIZER_H

#include <stdint.h>
#include "core/ps_coding/message_serializer.hh"
#include "examples/dense/echo_response_processor.hh"

namespace ps
{
namespace coding
{
namespace unittest
{
using namespace std;

template<typename T>
class EchoResponseSerializer : public MessageSerializer
{
public:
    EchoResponseSerializer(ps::coding::TemporaryBuffer&& t) : mTempBuffer(move(t))
    {}
    void Serialize() override
    {
        SetProcessorClassId(kEchoResponseProcessor_ClassId);
        AppendFragment(mTestMessage, mTestMessageLength);
    }

    const T* GetSourceData() const
    {
        return reinterpret_cast<const T*>(mTempBuffer.get());
    }
    int32_t GetSourceCount() const
    {
        return mTempBuffer.size() / sizeof(T);
    }
    void SetTestMessage(const char* message, int32_t length)
    {
        mTestMessage = message;
        mTestMessageLength = length;
    }

private:
    const char* mTestMessage;
    int32_t mTestMessageLength;
    ps::coding::TemporaryBuffer mTempBuffer;
};

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_SERIALIZER_H

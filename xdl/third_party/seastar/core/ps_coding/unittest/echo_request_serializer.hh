#ifndef ML_PS5_CODING_UNITTEST_ECHO_REQUEST_SERIALIZER_H
#define ML_PS5_CODING_UNITTEST_ECHO_REQUEST_SERIALIZER_H

#include "core/ps_coding/message_serializer.hh"
#include "echo_request_processor.hh"

namespace ps
{
namespace coding
{
namespace unittest
{

class EchoRequestSerializer : public MessageSerializer
{
public:
    void Serialize() override
    {
        SetProcessorClassId(kEchoRequestProcessor_ClassId);
        AppendFragment(mTestMessage, kTestMessageLength);
    }

    void SetTestMessage(const char* message) { mTestMessage = message; }

private:
    const char* mTestMessage = nullptr;
    static const int32_t kTestMessageLength = 128;
};

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_REQUEST_SERIALIZER_H

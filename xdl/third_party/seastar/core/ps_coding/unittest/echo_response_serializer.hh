#ifndef ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_SERIALIZER_H
#define ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_SERIALIZER_H

#include "core/ps_coding/message_serializer.hh"
#include "echo_response_processor.hh"

namespace ps
{
namespace coding
{
namespace unittest
{

class EchoResponseSerializer : public MessageSerializer
{
public:
    void Serialize() override
    {
        SetProcessorClassId(kEchoResponseProcessor_ClassId);
        AppendFragment(mTestMessage, kTestMessageLength);
    }

    void SetTestMessage(const char* message) { mTestMessage = message; }

private:
    const char* mTestMessage = nullptr;
    static const int kTestMessageLength = 128;
};

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_SERIALIZER_H

#ifndef ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_PROCESSOR_H
#define ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_PROCESSOR_H

#include "core/ps_coding/message_processor.hh"

namespace ps
{
namespace coding
{
namespace unittest
{

class EchoResponseProcessor : public MessageProcessor
{
public:
    seastar::future<> Process(ps::network::SessionContext* sc) override;
};

PS_DECLARE_MESSAGE_PROCESSOR_CLASS_ID(EchoResponseProcessor, 0xbf9ba9f2644d6da0);

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_RESPONSE_PROCESSOR_H

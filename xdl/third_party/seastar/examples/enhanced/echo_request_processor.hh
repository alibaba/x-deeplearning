#ifndef ML_PS5_CODING_UNITTEST_ECHO_REQUEST_PROCESSOR_H
#define ML_PS5_CODING_UNITTEST_ECHO_REQUEST_PROCESSOR_H

#include "core/ps_coding/message_processor.hh"

namespace ps
{
namespace coding
{
namespace unittest
{

class EchoRequestProcessor : public MessageProcessor
{
public:
    seastar::future<> Process(ps::network::SessionContext* sc) override;
};

PS_DECLARE_MESSAGE_PROCESSOR_CLASS_ID(EchoRequestProcessor, 0xcda34c53a7901d31);

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_CODING_UNITTEST_ECHO_REQUEST_PROCESSOR_H

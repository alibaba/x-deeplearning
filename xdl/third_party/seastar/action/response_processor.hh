#ifndef PS_ACTION_ACTION_RESPONSE_PROCESSOR_H
#define PS_ACTION_ACTION_RESPONSE_PROCESSOR_H

#include <stdint.h>
#include <iostream>
#include "core/ps_coding/message_processor.hh"

namespace ps
{
namespace action
{

class ActionResponseProcessor : public ps::coding::MessageProcessor
{
public:
    seastar::future<> Process(ps::network::SessionContext* sc) override;
private:
    ps::coding::TemporaryBuffer mTempBuffer;
};

PS_DECLARE_MESSAGE_PROCESSOR_CLASS_ID(ActionResponseProcessor, 0xbf9ba9f2644d6da1);

} // namespace action
} // namespace ps

#endif // PS_ACTION_ACTION_RESPONSE_PROCESSOR_H

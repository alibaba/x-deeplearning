#ifndef PS_ACTION_ACTION_REQUEST_PROCESSOR_H
#define PS_ACTION_ACTION_REQUEST_PROCESSOR_H

#include "core/ps_coding/message_processor.hh"

namespace ps
{
namespace action
{
using namespace std;

class ActionRequestProcessor : public ps::coding::MessageProcessor
{
public:
    seastar::future<> Process(ps::network::SessionContext* sc) override;
};

PS_DECLARE_MESSAGE_PROCESSOR_CLASS_ID(ActionRequestProcessor, 0xcda34c53a7901d31);

} // namespace action
} // namespace ps

#endif // PS_ACTION_ACTION_REQUEST_PROCESSOR_H

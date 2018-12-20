#include <stdint.h>
#include <iostream>
#include "core/ps_coding/message_processor_factory.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "core/reactor.hh"
#include "service/client.hh"
#include "service/client_network_context.hh"
#include "action/utility.hh"
#include "action/response_processor.hh"
#include "action/push_response_item.hh"

namespace ps
{
namespace action
{
using namespace std;

seastar::future<> ActionResponseProcessor::Process(ps::network::SessionContext* sc)
{
    this->GetMessageHeader().mTimeStamp.mWorkerProcessTime
        = utility::CoutCurrentTimeOnNanoSec("______worker_process_time", false);
    ps::network::PsWorkItem* item = new ps::action::PushResponseItem(this->GetMessageHeader().mTimeStamp);
    ps::network::ClientConnection* clientConn
        = dynamic_cast<ps::network::ClientConnection *>(sc->GetConnection());
    if (clientConn == NULL)
    {
        assert("Serious Error. Connect is NULL.");
    }
    item->SetSideSequ(this->GetMessageHeader().mSequence);
    clientConn->mClient.GetClientNetworkContext()->PushItemBack(item,
            seastar::engine().cpu_id(), this->GetMessageHeader().mUserThreadId);
    return seastar::make_ready_future<>();
}

PS_REGISTER_MESSAGE_PROCESSOR(ActionResponseProcessor);

} // namespace action
} // namespace ps

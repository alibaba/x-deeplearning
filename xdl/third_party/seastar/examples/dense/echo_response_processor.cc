#include <stdint.h>
#include "examples/dense/echo_response_processor.hh"
#include "core/ps_coding/message_processor_factory.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "service/client.hh"
#include "service/client_network_context.hh"
#include "examples/dense/worker_response_item.hh"

uint64_t CoutCurrentTimeOnNanoSec(const char * prompt, bool output);

namespace ps
{
namespace coding
{
namespace unittest
{
using namespace std;

seastar::future<> EchoResponseProcessor::Process(ps::network::SessionContext* sc)
{
    this->GetMessageHeader().mTimeStamp.mWorkerProcessTime
        = CoutCurrentTimeOnNanoSec("______worker_process_time", false);
    ps::network::PsWorkItem* pItem
        = new ps::coding::unittest::WorkerResponseItem<double>(
                move(GetDataBuffer()), this->GetMessageHeader().mTimeStamp);
    ps::network::ClientConnection * clientConn
        = dynamic_cast<ps::network::ClientConnection *>(sc->GetConnection());
    if (clientConn == NULL)
    {
        assert("Serious error！");
    }
    pItem->SetSideSequ(this->GetMessageHeader().mSequence);

    clientConn->mClient.GetClientNetworkContext()->PushItemBack(pItem,
        seastar::engine().cpu_id(), this->GetMessageHeader().mUserThreadId);

    return seastar::make_ready_future<>();
}

PS_REGISTER_MESSAGE_PROCESSOR(EchoResponseProcessor);

} // namespace unittest
} // namespace coding
} // ps

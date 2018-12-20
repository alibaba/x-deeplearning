#include <iostream>
#include "examples/dense/echo_request_processor.hh"
#include "examples/dense/echo_response_serializer.hh"
#include "examples/dense/echo_response_item.hh"
#include "core/reactor.hh"
#include "core/ps_coding/message_processor_factory.hh"
#include "core/ps_queue_hub/queue_hub.hh"

uint64_t CoutCurrentTimeOnNanoSec(const char * prompt, bool output);

namespace ps
{
namespace coding
{
namespace unittest
{
using namespace std;

seastar::future<> EchoRequestProcessor::Process(ps::network::SessionContext* sc)
{
    EchoResponseSerializer<double>* serializer
        = new EchoResponseSerializer<double>(move(GetDataBuffer()));
    *serializer->GetTimeStamp() = this->GetMessageHeader().mTimeStamp;
    serializer->GetTimeStamp()->mServerProcessTime
        = CoutCurrentTimeOnNanoSec("______server_process_time", false);
    serializer->SetSequence(this->GetMessageHeader().mSequence);
    serializer->SetUserThreadId(this->GetMessageHeader().mUserThreadId);
    ps::network::PsWorkItem* item
        = new ps::coding::unittest::EchoResponseItem<double>(0, sc, serializer);

    //item->Run();
    //return item->Complete();

    const int32_t shardId = *reinterpret_cast<const int32_t*>(GetMetaBuffer().get());
    std::pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr
        = ps::network::QueueHubFactory::GetInstance().GetHubWithoutLock<ps::network::Item>("SEASTAR");
    ps::network::QueueHub<ps::network::Item>* outputHub = qhr.second;
    outputHub->Enqueue(item, seastar::engine().cpu_id(), shardId, NULL);
    return seastar::make_ready_future<>();
}

PS_REGISTER_MESSAGE_PROCESSOR(EchoRequestProcessor);

} // namespace unittest
} // namespace coding
} // ps

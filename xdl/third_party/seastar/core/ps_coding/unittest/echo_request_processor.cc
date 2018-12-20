#include "echo_request_processor.hh"
#include "echo_response_serializer.hh"
#include "core/ps_coding/message_processor_factory.hh"
#include "core/ps_coding/unittest/echo_response_item.hh"
#include "core/ps_queue_hub/queue_hub.hh"
#include "core/reactor.hh"
#include <iostream>

using namespace std;

namespace ps
{
namespace coding
{
namespace unittest
{

seastar::future<> EchoRequestProcessor::Process(ps::network::SessionContext* sc)
{
    EchoResponseSerializer* serializer = new EchoResponseSerializer;
    serializer->SetSequence(this->GetMessageHeader().mSequence);
    serializer->SetUserThreadId(this->GetMessageHeader().mUserThreadId);
    serializer->SetCalculateCostTime(this->GetMessageHeader().mCalculateCostTime);

    static std::string respData(127, 'X');
    serializer->SetTestMessage(respData.c_str());

/*
    // Method without queue
    //
    return sc->Write(serializer);

*/
    // Method with queue
    //
    
    ps::network::PsWorkItem* item
        = new ps::coding::unittest::EchoResponseItem(0, sc, serializer);
    //std::pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr
    //    = ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>(queueName);
    std::pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr
        = ps::network::QueueHubFactory::GetInstance().GetDefaultQueueForServer();

    ps::network::QueueHub<ps::network::Item>* outputHub = qhr.second;
    // In seperate server_client mode
    // client cpu range: [0,smp::client_count)
    // server cpu range: [smp::client_count,smp::count)
    unsigned queueId = seastar::engine().cpu_id();
    if (seastar::smp::seperate_server_client)
    {
        // cpu id to queue index
        queueId -= seastar::smp::client_count;
    }
    outputHub->Enqueue(item, queueId, 0, NULL);

    return seastar::make_ready_future<>();
}

PS_REGISTER_MESSAGE_PROCESSOR(EchoRequestProcessor);

} // namespace unittest
} // namespace coding
} // ps

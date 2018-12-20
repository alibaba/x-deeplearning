#include <stdint.h>
#include "echo_response_processor.hh"
#include "core/ps_coding/message_processor_factory.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "service/client.hh"
#include "service/client_network_context.hh"

#include <iostream>
using namespace std;

namespace ps
{
namespace coding
{
namespace unittest
{

seastar::future<> EchoResponseProcessor::Process(ps::network::SessionContext* sc)
{
    // exception !
    if (sc->GetIsBadStatus())
    {
        cout << "Bad session context status, and handle exception in processor ! ! !" << endl;

        // get all relative sequence
        const vector<uint64_t>& sequences = sc->GetSavedSequenceWhenBroken();
        for (unsigned i = 0; i < sequences.size(); ++i)
        {
            cout << sequences[i] << "   ";
        }
        cout << endl;

        // TODO: handle exception
        return seastar::make_ready_future<>();
    }

    // exist timeout packets
    if (sc->GetHaveTimeOutPacket())
    {
        const vector<uint64_t>& sequences = sc->GetSavedSequenceWhenBroken();
        for (unsigned i = 0; i < sequences.size(); ++i)
        {
            cout << sequences[i] << "   ";
        }
        cout << endl;

        // TODO: handle timeout packets
        return seastar::make_ready_future<>();
    }

    // else: send item to user
     
    ps::network::ClientConnection * clientConn
        = dynamic_cast<ps::network::ClientConnection *>(sc->GetConnection());
    if (clientConn == NULL)
    {
        assert("Serious error！");
    }
    ps::network::PsWorkItem* pItem = new ps::network::PsWorkItem();

    pItem->SetSideSequ(this->GetMessageHeader().mSequence);

    unsigned queueId = engine().cpu_id();
    // user thread id, pItem will be delete in f.Get()
    clientConn->mClient.GetClientNetworkContext()->PushItemBack(pItem,
        queueId, this->GetMessageHeader().mUserThreadId);

    return seastar::make_ready_future<>();
}

PS_REGISTER_MESSAGE_PROCESSOR(EchoResponseProcessor);

} // namespace unittest
} // namespace coding
} // ps

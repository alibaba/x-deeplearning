#include "seastar_session_context.hh"
#include "net/packet.hh"
#include "core/deleter.hh"
#include "session_context.hh"
#include "base_connection.hh"
#include "client_network_context.hh"
#include "client.hh"
#include "server.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "core/ps_coding/message_processor.hh"
#include "core/ps_coding/message_processor_factory.hh"
#include <vector>
#include <errno.h>
#include <assert.h>
#include <iostream>

using namespace std;

namespace ps
{
namespace network
{

SeastarSessionContext::~SeastarSessionContext()
{
    // conn will be delete by the session delete
    if (mConn && (mNetworkContext == NULL || (mNetworkContext != NULL && mNetworkContext->GetRunning())))
    {
        delete mConn;
        mConn = NULL;
    }
}

void SeastarSessionContext::ReportError(int64_t serverId, int64_t userThreadId, uint64_t seq,
        ps::network::ConnectionStatus status, int64_t eno)
{
    // client
    ClientConnection* clientConn = dynamic_cast<ClientConnection *>(this->mConn);
    if (this->mConn->GetRoleType() == RoleType::Client && clientConn != NULL)
    {
        // There are two methods to handle the exception:
        // 1). If user set the response processor id, 
        //     then we should handle exception in the processor.
        // 2). In general, we send the exception back via the MxN queue.
        if (GetProcessorId() == 0)
        {
            cout << "report error: serverid=" << serverId
                << " ,userThreadId=" << userThreadId << " ,seq=" << seq
                << " ,status=" << status << " ,eno=" << eno << endl;
            ps::network::PsWorkItem* pItem = new ps::network::PsWorkItem();
            pItem->SetSideSequ(seq);
            pItem->SetStatus(status);
            pItem->SetErrno(eno);
            pItem->SetServerId(serverId);
            unsigned queueId = engine().cpu_id();
            clientConn->mClient.GetClientNetworkContext()->PushItemBack(pItem, queueId, userThreadId);
        }
        else
        {
            std::cout << "report error in process, seq=" << seq << ", serverid=" << serverId << std::endl;
            // set bad connection(status)
            SetBadStatus(true);
            // the saved seqence id will be used in processor
            ClearSavedSequence();
            PutOneToSavedSequence(seq);
            ps::coding::MessageProcessor* processor =
                ps::coding::MessageProcessorFactory::GetInstance().
                CreateInstance(GetProcessorId());
            processor->Process(this).then([processor] () {
                // ? here
                //SetBadStatus(false);
                delete processor;
            });
        }
    }
}

seastar::future<> SeastarSessionContext::Write(ps::coding::MessageSerializer* serializer)
{
    // client do this, server connection id -1(default, need be refine)
    if (this->mConn->GetConnStatus() == false && this->mConn->GetRoleType() == RoleType::Client)
    {
        // connection has been closed, return an exception item
        // notify the user thread when invoke f.Get()
        ReportError(this->mConn->GetConnId(), serializer->GetUserThreadId(), serializer->GetSequence(),
                ps::network::ConnectionStatus::BadConnection, ECONNRESET);
        return seastar::make_ready_future<>();
    }

    return DoWrite(serializer).then_wrapped([this, serializer] (auto&& f)
    {
        try
        {
            f.get();
            return seastar::make_ready_future<>();
        }
        catch (std::exception& ex)
        {
            std::cout << "write exception: " << ex.what() << ", errno: " << errno << std::endl;
            ps::network::ConnectionStatus status = ps::network::ConnectionStatus::Unknow;
            if (ps::network::IsConnectionBroken(errno))
            {
                this->mConn->SetConnStatus(false);
                status = ps::network::ConnectionStatus::BadConnection;
            }
            this->ReportError(this->mConn->GetConnId(), serializer->GetUserThreadId(),
                    serializer->GetSequence(), status, errno);
            return seastar::make_ready_future<>();
        }
    });
}

seastar::future<> SeastarSessionContext::DoWrite(ps::coding::MessageSerializer* serializer)
{
#ifdef USE_STATISTICS
    // For RTT test
    ServerConnection* serverConn = dynamic_cast<ServerConnection *>(this->mConn);
    if (serverConn)
    {
        // Just store the cost of process
        serializer->SetCalculateCostTime(GetCurrentTimeInUs() - serializer->GetCalculateCostTime());
    }
#endif

    std::vector<seastar::net::fragment> fragments(serializer->GetFragments());
    seastar::net::packet pack(std::move(fragments), seastar::make_deleter(seastar::deleter(), [serializer]
                { delete serializer;}));

    uint64_t t = GetCurrentTimeInUs();
    uint64_t seq = serializer->GetSequence();
    int64_t userThreadId = serializer->GetUserThreadId();

    // save to client map before write
    if (this->mConn->GetRoleType() == RoleType::Client)
    {
        ClientConnection* clientConn = dynamic_cast<ClientConnection *>(this->mConn);
        if (clientConn == NULL)
        {
            assert("do_write: bad connection! ");
        }
        clientConn->InsertRequestInfoToMap(seq, userThreadId, t);
    }

    return mConn->mOut.write(std::move(pack)).then([this]
    {
        return this->mConn->mOut.flush().then([]
        {
            // Nothing
            return seastar::make_ready_future<>();
        });
    });
}

seastar::future<> SeastarSessionContext::Reconnect(std::string serverAddr, int64_t remoteId, int64_t userThreadId, uint64_t sequenceId)
{
    ClientConnection* clientConn = dynamic_cast<ClientConnection *>(this->mConn);
    if (clientConn == NULL)
    {
        assert("Reconnect: bad connection!");
    }
    return clientConn->mClient.Connect(serverAddr, remoteId, userThreadId, sequenceId, true);
}

} // namespace network
} // namespace ps


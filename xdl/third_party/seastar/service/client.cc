#include "client.hh"
#include "core/ps_queue_hub/queue_hub.hh"
#include "core/ps_queue_hub/queue_work_item.hh"
#include "net/packet.hh"
#include "core/ps_coding/message_processor.hh"
#include "core/ps_coding/message_processor_factory.hh"
#include "service/seastar_session_context.hh"
#include "service/client_network_context.hh"
#include "service/base_connection.hh"
#include "service/queue_poller.hh"
#include "service/seastar_status.hh"
#include "service/connect_work_item.hh"
#include <errno.h>
#include <iostream>

using namespace std;
using namespace ps;
using namespace ps::coding;

namespace ps
{
namespace network
{

seastar::future<> ClientConnection::CloseIn()
{
    if (mIsInClose)
    {
       return seastar::make_ready_future<>();
    }
    return mIn.close();
}

seastar::future<> ClientConnection::CloseOut()
{
    if (mIsOutClose)
    {
       return seastar::make_ready_future<>();
    }
    return mOut.close();
}

ClientNetworkContext* ClientConnection::GetClientNetworkContext()
{
    return mClient.GetClientNetworkContext();
}

void ClientConnection::InsertRequestInfoToMap(uint64_t sequenceId, int64_t userThreadId, uint64_t t)
{
    RequestMetaInfo info{sequenceId, userThreadId, t};
    mRequestInfoSet.insert(std::move(info));
}

//void ClientConnection::EraseRequestInfoFromMap(uint64_t sequenceId, int64_t userThreadId)
void ClientConnection::EraseRequestInfoFromMap(ps::coding::MessageHeader& header)
{
    // here we dont need sendTime
    RequestMetaInfo info{header.mSequence, header.mUserThreadId, 0};
    std::unordered_set<RequestMetaInfo, RequestMetaInfoHashFunc, RequestMetaInfoCmp>::const_iterator ele
         = mRequestInfoSet.find(info);
    if (ele != mRequestInfoSet.end())
    {
#ifdef USE_STATISTICS
        uint32_t totalRtt = (uint32_t)(GetCurrentTimeInUs() - ele->sendTime);
        uint32_t curRtt = totalRtt - (uint32_t)header.mCalculateCostTime;

        //std::cout << "Total = " << totalRtt << ", process_cost = " << header.mCalculateCostTime << ", rtt = " << curRtt << std::endl;

        mClient.Stats().mMinRtt = std::min(mClient.Stats().mMinRtt, curRtt);
        mClient.Stats().mMaxRtt = std::max(mClient.Stats().mMaxRtt, curRtt);
        mClient.Stats().mReqCount++;
        mClient.Stats().mTotalRtt += totalRtt;
#endif
        mRequestInfoSet.erase(ele);
    }
    else
    {
        std::cout << "Miss packet: sequence = " << header.mSequence << ", user thread id = " << header.mUserThreadId << std::endl;
    }
}

void ClientConnection::ResponseAllWhenDisConnect(int32_t eno)
{
    std::cout << "ResponseAllWhenDisConnect ..." << std::endl;

    // will handle the exception in processor
    uint64_t processorId = mClient.GetClientNetworkContext()->GetResponseProcessorIdOfServer(GetConnId());
    if (processorId != 0)
    {
        SessionContext* sc = mClient.GetClientNetworkContext()->GetSessionOfId(GetConnId());
        sc->SetBadStatus(true);
        // set saved sequence, will be used in processor
        sc->ClearSavedSequence();
        for(auto iter = mRequestInfoSet.begin(); iter != mRequestInfoSet.end(); ++iter)
        {
            sc->PutOneToSavedSequence(iter->sequenceId);
        }
        ps::coding::MessageProcessor* processor =
            ps::coding::MessageProcessorFactory::GetInstance().
            CreateInstance(processorId);
        processor->Process(sc).then([processor] ()
        {
            // ? here
            // sc->SetBadStatus(false);
            delete processor;
        });
    }
    // will send items back to user
    else
    {
        for(auto iter = mRequestInfoSet.begin(); iter != mRequestInfoSet.end(); ++iter)
        {
            ps::network::PsWorkItem* pItem = new ps::network::PsWorkItem();
            pItem->SetSideSequ(iter->sequenceId);
            pItem->SetStatus(ps::network::ConnectionStatus::BadConnection);
            pItem->SetErrno(eno);
            pItem->SetServerId(GetConnId());
            unsigned queueId = engine().cpu_id();
            this->mClient.GetClientNetworkContext()->PushItemBack(pItem, queueId, (*iter).userThreadId);
        }
    }
    mRequestInfoSet.clear();
}

//-----------------------------------------------------------

//
// The function can ONLY be called by other(user) thread(not seastar thread),
// here will return the item for MxN queue.
// If you want to connect to server in seastar, you should use:
// seastar::future<int> SeastarClient::ConnectToOne(int64_t connId, std::string serverAddr, bool isReconnect)
//
seastar::future<> SeastarClient::Connect(std::string serverAddr,
        int64_t connId, int64_t userThreadId, uint64_t seq, bool isReconnect)
{
    return ConnectToOne(connId, serverAddr, isReconnect).then([this, serverAddr, connId, userThreadId, seq, isReconnect] (auto code)
    {
        // pItem will be delete in f.Get()
        ps::network::PsWorkItem* pItem = NULL;
        if (isReconnect)
        {
            pItem = new ps::network::ConnectWorkItem(true);
        }
        else
        {
            pItem = new ps::network::ConnectWorkItem(false);
        }
        //new ps::network::PsWorkItem();
        pItem->SetSideSequ(seq);
        // can't connect to server
        if (code == -1)
        {
            pItem->SetStatus(CanNotConnectTo);
        }
        unsigned queueId = engine().cpu_id();
        this->mClientNetworkContext->PushItemBack(pItem, queueId, userThreadId);

        return seastar::make_ready_future();
    });
}

ClientNetworkContext* SeastarClient::GetClientNetworkContext()
{
    return mClientNetworkContext;
}

seastar::future<int> SeastarClient::ConnectToOne(int64_t connId, std::string serverAddr, bool isReconnect)
{
    // The first try to connect to a server
    if (mSynSentStartTime == 0)
    {
        struct timeval tv;
        gettimeofday(&tv,NULL);
        mSynSentStartTime = tv.tv_sec * 1000000 + tv.tv_usec;
    }

    return seastar::engine().net().connect(seastar::make_ipv4_address(seastar::ipv4_addr{serverAddr}))
        .then_wrapped([connId, serverAddr, isReconnect, this] (auto&& f)
    {
        try
        {
            seastar::connected_socket sk = std::move(get<0>(f.get()));

            if (this->GetTcpNoDelayOn())
            {
                // set tcp no delay, decrease the rtt
                // solve the 40ms delay problem caused by lazy(delay) ack
                sk.set_nodelay(true);
            }

            if (this->GetTcpKeepAliveIdle() > 0) {
                seastar::net::tcp_keepalive_params keep_alive;
                keep_alive.idle = std::chrono::seconds(this->GetTcpKeepAliveIdle());
                keep_alive.interval = std::chrono::seconds(this->GetTcpKeepAliveInterval());
                keep_alive.count = this->GetTcpKeepAliveCnt();
                sk.set_keepalive(true);
                sk.set_keepalive_parameters(keep_alive);
            }

            this->Start(connId, std::move(sk), serverAddr);

            if (isReconnect)
            {
                seastar::print("Reconnect connection: server_id = %3d on cpu %3d, server addr: %s.\n",
                        connId, seastar::engine().cpu_id(), serverAddr);
            }
            else
            {
                mConnConnected++;
                seastar::print("Established connection: server_id = %3d on cpu %3d, server addr: %s.\n",
                        connId, seastar::engine().cpu_id(), serverAddr);
            }
            //seastar::print("Current connection count %3d on cpu %3d.\n", mConnConnected,
            //        seastar::engine().cpu_id());

            mSynSentStartTime = 0;
            return seastar::make_ready_future<int>(0);
        }
        catch (std::exception& ex)
        {
            std::cout << "server addr: " << serverAddr << ", connect failed: "
                << ex.what() << " errno:" << errno << std::endl;

            struct timeval tv;
            gettimeofday(&tv,NULL);
            // 10s connect timeout
            if (tv.tv_sec * 1000000 + tv.tv_usec - mSynSentStartTime >= 10*1000*1000)
            {
                std::cerr << "server: " << serverAddr << " may not be launched." << std::endl;
                mSynSentStartTime = 0;
                return seastar::make_ready_future<int>(-1);
            }

            return this->ConnectToOne(connId, serverAddr, isReconnect);
        }
    });
}

seastar::future<seastar::stop_iteration> ReadAndProcess(BaseConnection* conn)
{
    return conn->mIn.read_exactly(sizeof(MessageHeader)).then([conn] (auto&& header)
    {
        if (header.empty())
        {
            return conn->mIn.close().then([conn]
            {
                return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::yes);
            });
        }

        const ps::coding::MessageHeader& messageHeader =
            *reinterpret_cast<const ps::coding::MessageHeader*>(
                header.get());
//std::cout << "read size: " << sizeof(MessageHeader) << ", Header size: "  << header.size()  << ", Header: " << messageHeader.mProcessorClassId << ":" << messageHeader.mSequence << ":" << messageHeader.mUserThreadId << std::endl;

        ps::coding::MessageProcessor* processor =
            ps::coding::MessageProcessorFactory::GetInstance().CreateInstance(
                messageHeader.mProcessorClassId);
        processor->GetMessageHeader() = messageHeader;

        const size_t metaBufferSize = processor->GetMessageHeader().mMetaBufferSize;
        return conn->mIn.read_exactly(metaBufferSize).then([conn, processor, metaBufferSize] (auto&& meta)
        {
            if (meta.empty() && metaBufferSize > 0)
            {
                return conn->mIn.close().then([conn]
                {
                    return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::yes);
                });
            }

            processor->GetMetaBuffer() = std::move(meta);

            const size_t dataBufferSize = processor->GetMessageHeader().mDataBufferSize;
            return conn->mIn.read_exactly(dataBufferSize).then([conn, processor, dataBufferSize] (auto&& body)
            {
                if (body.empty() && dataBufferSize > 0)
                {
                    return conn->mIn.close().then([conn]
                    {
                        return seastar::make_ready_future<seastar::stop_iteration>(
                                seastar::stop_iteration::yes);
                    });
                }

                //std::cout << "client recv: " << body.get() << std::endl;
                processor->GetDataBuffer() = std::move(body);
                return processor->Process(conn->GetSessionContext()).then([processor, conn] ()
                {
                    //std::cout << "client delete processor..." << std::endl;

                    // in client side: erase the element in the request set
                    if (conn->GetRoleType() == RoleType::Client)
                    {
                        ClientConnection* clientConn = dynamic_cast<ClientConnection *>(conn);
                        if (clientConn == NULL)
                        {
                            assert("read successful: bad connection! ");
                        }
                        clientConn->EraseRequestInfoFromMap(processor->GetMessageHeader());
                    }

                    delete processor;
                    return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::no);
                });
           });
       });
    });
}

void SeastarClient::Start(int64_t connId, seastar::connected_socket&& sk, const std::string& serverAddr)
{
    auto conn = new ClientConnection(RoleType::Client, std::move(sk), *this, connId, serverAddr,
            mTimeoutInterval);
    SessionContext* sc = new SeastarSessionContext(conn, mClientNetworkContext);
    unique_ptr<SessionContext> usc(sc);
    mClientNetworkContext->SetSessionContextOfId(std::move(usc), connId);
    conn->SetSessionContext(sc);

    //seastar::engine().GetQueuePoller().Poll();
    seastar::repeat([conn, this] ()
    {
        return ReadAndProcess(conn);
    }).then_wrapped([conn, this] (auto&& f)
    {
        try
        {
            std::cout << "read END!" << std::endl;
            // server close the socket, it's also an error
            f.get();
        }
        catch(std::exception& ex)
        {
            std::cout << "read Exception!  " << ex.what() << ", errno: " << errno  << std::endl;
        }

        // client should response to all：connection is broken
        if (conn->GetRoleType() == RoleType::Client)
        {
            ClientConnection* clientConn = dynamic_cast<ClientConnection *>(conn);
            if (clientConn == NULL)
            {
                assert("read failed: bad connection! ");
            }
            clientConn->ResponseAllWhenDisConnect(errno);
        }

        std::cout << "read: set connection status false..." << std::endl;
        // server close the socket or exception
        // set the conn_status to failed
        conn->SetConnStatus(false);
    });
}

} // namespace network
} // namespace ps

#include "service/server.hh"
#include "net/packet.hh"
#include "core/ps_coding/message_processor_factory.hh"
#include "core/ps_coding/message_processor.hh"
#include "core/ps_coding/unittest/echo_request_processor.hh"
#include "service/queue_poller.hh"
#include <errno.h>
#include <iostream>

using namespace std;

namespace ps
{
namespace network
{

SeastarServer::SeastarServer(uint16_t port, bool tcpNoDelayOn, int tcpKeepAliveIdle, int tcpKeepAliveCnt, int tcpKeepAliveInterval)
    : mPort(port), mTcpNoDelayOn(tcpNoDelayOn), mTcpKeepAliveIdle(tcpKeepAliveIdle), mTcpKeepAliveCnt(tcpKeepAliveCnt), mTcpKeepAliveInterval(tcpKeepAliveInterval) { }

SeastarServer::~SeastarServer() { }

void SeastarServer::Start()
{
    seastar::listen_options lo;
    lo.reuse_address = true;
    mListener = seastar::engine().listen(seastar::make_ipv4_address({mPort}), lo);

    // Poll() will be start in reactor
    //seastar::engine().GetQueuePoller().Poll();

    // accept and read
    seastar::keep_doing([this]
    {
        return mListener->accept().then_wrapped([this] (auto&& f) mutable
        {
            try
            {
                std::tuple<seastar::connected_socket, seastar::socket_address> data = f.get();
                seastar::connected_socket fd = std::move(get<0>(data));
                if (this->GetTcpNoDelayOn())
                {
                    fd.set_nodelay(true);
                }

                if (this->GetTcpKeepAliveIdle() > 0) {
                    seastar::net::tcp_keepalive_params keep_alive;
                    keep_alive.idle = std::chrono::seconds(this->GetTcpKeepAliveIdle());
                    keep_alive.interval = std::chrono::seconds(this->GetTcpKeepAliveInterval());
                    keep_alive.count = this->GetTcpKeepAliveCnt();
                    fd.set_keepalive(true);
                    fd.set_keepalive_parameters(keep_alive);
                }

                seastar::socket_address addr = get<1>(data);
                auto conn = new ServerConnection(RoleType::Server, std::move(fd), addr, *this);

                this->Process(conn).finally([conn]
                {
                    return conn->mOut.close().finally([conn]
                    {
                        //cout << "client disconnect..." << endl;
                        conn->DeleteSessionContext();
                    });
                });
           }
           catch (std::exception& ex)
           {
               cout << "accept error: " << ex.what() << " , errno:" << errno << endl;
           }
        });
    }).or_terminate();
}

seastar::future<> SeastarServer::Process(ServerConnection* conn)
{
    return seastar::repeat([conn, this] ()
    {
        return this->ReadAndProcess(conn);
    }).then_wrapped([conn, this] (auto&& f)
    {
        try
        {
            f.get();
            cout << "server read end: client close the connection." << endl;
        }
        catch(std::exception& ex)
        {
            cout << "server read exception: " << ex.what()
            << ", errno: " << errno << endl;
        }
    });
}

seastar::future<seastar::stop_iteration> SeastarServer::ReadAndProcess(ps::network::BaseConnection* conn)
{
    return conn->mIn.read_exactly(sizeof(ps::coding::MessageHeader)).then([conn] (auto&& header)
    {
        //cout << "server receive one request" << endl;
        if (header.empty())
        {
            return conn->mIn.close().then([conn]
            {
                return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::yes);
            });
        }

        // statistic qps
        ServerConnection* serverConn = dynamic_cast<ServerConnection *>(conn);
        if (serverConn) ++serverConn->mServer.Stats().mEcho;

        const ps::coding::MessageHeader& messageHeader
            = *reinterpret_cast<const ps::coding::MessageHeader*>(header.get());

//std::cout << "read size: " << sizeof(ps::coding::MessageHeader) << ", Header size: "  << header.size()  << ", Header: " << messageHeader.mProcessorClassId << ":" << messageHeader.mSequence <<  ":" << messageHeader.mUserThreadId << std::endl;

        ps::coding::MessageProcessor* processor
            = ps::coding::MessageProcessorFactory::GetInstance().CreateInstance(
                messageHeader.mProcessorClassId);
        processor->GetMessageHeader() = messageHeader;

        const size_t metaBufferSize = processor->GetMessageHeader().mMetaBufferSize;
        return conn->mIn.read_exactly(metaBufferSize).then([conn, processor] (auto&& meta)
        {
            processor->GetMetaBuffer() = std::move(meta);
            const size_t dataBufferSize = processor->GetMessageHeader().mDataBufferSize;
            return conn->mIn.read_exactly(dataBufferSize).then([conn, processor] (auto&& body)
            {
                //cout << "server recv: " << body.get() << endl;
                processor->GetDataBuffer() = std::move(body);
            #ifdef USE_STATISTICS
                // For RTT test: record the starting ts
                processor->GetMessageHeader().mCalculateCostTime = GetCurrentTimeInUs();
            #endif
                return processor->Process(conn->GetSessionContext()).then([processor] ()
                {
                    delete processor;
                    return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::no);
                });
            });
        });
    });
}

} // namespace network
} // namespace ps


#ifndef SERVICE_SERVER_HH_
#define SERVICE_SERVER_HH_

#include "core/future-util.hh"
#include "core/shared_ptr.hh"
#include "core/reactor.hh"
#include "net/api.hh"
#include "core/ps_coding/message_serializer.hh"
#include "service/base_connection.hh"
#include "service/seastar_session_context.hh"
#include "stats_printer.hh" 
#include "service/common.hh"

namespace ps
{
namespace network
{

class ServerConnection;

class SeastarServer
{
private:
    seastar::lw_shared_ptr<seastar::server_socket> mListener;
    uint16_t mPort;

    // for statstics
    SystemStats mSystemStats;
public:
    SeastarServer(uint16_t port, bool tcpNoDelayOn, int tcpKeepAliveIdle, int tcpKeepAliveCnt, int tcpKeepAliveInterval);
    ~SeastarServer();

    void Start();

    seastar::future<> Process(ServerConnection*);

    // the func will be called by seastar-core when finished
    // the func name must be 'stop', you can NOT use 'Stop'
    seastar::future<> stop()
    {
        return seastar::make_ready_future<>();
    }

    SystemStats& Stats()
    {
        return mSystemStats;
    }

    bool GetTcpNoDelayOn() const
    {
        return mTcpNoDelayOn;
    }

    int GetTcpKeepAliveIdle() const
    {
        return mTcpKeepAliveIdle;
    }

    int GetTcpKeepAliveCnt() const
    {
        return mTcpKeepAliveCnt;
    }

    int GetTcpKeepAliveInterval() const
    {
        return mTcpKeepAliveInterval;
    }
private:
    seastar::future<seastar::stop_iteration> ReadAndProcess(ps::network::BaseConnection* conn);

    // whether enable tcp_no_delay flag in socket(connection)
    bool mTcpNoDelayOn;
    int mTcpKeepAliveIdle;
    int mTcpKeepAliveCnt;
    int mTcpKeepAliveInterval;
};


class ServerConnection : public BaseConnection
{
public:
    seastar::socket_address mAddr;
    SeastarServer& mServer;

    ServerConnection(RoleType role, seastar::connected_socket&& socket, seastar::socket_address addr, SeastarServer& s)
        : BaseConnection(role, std::move(socket))
        , mAddr(addr), mServer(s)
    {
        //mOut.set_batch_flushes(false);
        mSessionContext = new SeastarSessionContext(this, NULL);
        ServerConnctionManager::RecordConnection(mSessionContext);

#ifdef USE_STATISTICS
        ++mServer.Stats().mCurrConnections;
        ++mServer.Stats().mTotalConnections;
#endif
	}

    ~ServerConnection()
    {
#ifdef USE_STATISTICS
        --mServer.Stats().mCurrConnections;
#endif
	}
};

} // namespace network
} // namespace ps

#endif /* SERVICE_SERVER_HH_ */

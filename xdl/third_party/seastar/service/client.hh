#ifndef SERVICE_CLIENT_HH_
#define SERVICE_CLIENT_HH_

#include "core/future-util.hh"
#include "core/ps_coding/message_serializer.hh"
#include "base_connection.hh"
#include "stats_printer.hh"
#include "core/ps_coding/message_header.hh"
#include "service/packet_timer.hh"
#include <unordered_set>
#include <vector>

namespace ps
{
namespace network
{

class ClientNetworkContext;
class SeastarClient;

struct RequestMetaInfo
{
    uint64_t sequenceId;
    int64_t  userThreadId;
    uint64_t sendTime;
};

struct RequestMetaInfoCmp
{
    bool operator()(const struct RequestMetaInfo& x, const struct RequestMetaInfo& y) const
    {
        // dont need to compare sendTime
        return (y.sequenceId == x.sequenceId) && (y.userThreadId == x.userThreadId);
    }
};

struct RequestMetaInfoHashFunc
{
    size_t operator()(const struct RequestMetaInfo& r) const
    {
        return std::hash<int>()(r.sequenceId);
    }
};

//-----------------------------------------

class ClientConnection : public BaseConnection
{
public:
    bool mIsInClose;
    bool mIsOutClose;
    SeastarClient& mClient;
    std::string mServerAddr;

    ClientConnection(RoleType role, seastar::connected_socket&& fd, SeastarClient& c,
            uint64_t id, std::string saddr, uint64_t inerval)
        : BaseConnection(role, std::move(fd), id), mClient(c), mServerAddr(saddr)
        , mPacketTimer(this, inerval)
    {
        mIsInClose = false;
        mIsOutClose = false;

        // start packet timetout timer
        mPacketTimer.Start();
    }

    ~ClientConnection() { }

    seastar::future<> CloseIn();
    seastar::future<> CloseOut();

    void InsertRequestInfoToMap(uint64_t sequenceId, int64_t userThreadId, uint64_t t);
    //void EraseRequestInfoFromMap(uint64_t sequenceId, int64_t userThreadId);
    void EraseRequestInfoFromMap(ps::coding::MessageHeader& header);
    void ResponseAllWhenDisConnect(int32_t);

    ClientNetworkContext* GetClientNetworkContext();

    SeastarClient& GetClient()
    {
        return mClient;
    }

    std::unordered_set<RequestMetaInfo, RequestMetaInfoHashFunc, RequestMetaInfoCmp>* GetRequestInfoSet()
    {
        return &mRequestInfoSet;
    }

private:
    PacketTimer mPacketTimer;
    std::unordered_set<RequestMetaInfo, RequestMetaInfoHashFunc, RequestMetaInfoCmp> mRequestInfoSet;
};

//-----------------------------------------

class SeastarClient
{
public:
    SeastarClient(ClientNetworkContext* networkContext, bool tcpNoDelayOn, int tcpKeepAliveIdle, int tcpKeepAliveCnt, int tcpKeepAliveInterval, uint64_t interval = 1000)
        : mClientNetworkContext(networkContext), mTcpNoDelayOn(tcpNoDelayOn), mTcpKeepAliveIdle(tcpKeepAliveIdle), mTcpKeepAliveCnt(tcpKeepAliveCnt), mTcpKeepAliveInterval(tcpKeepAliveInterval), mTimeoutInterval(interval)
    {
        mConnConnected = 0;
        mSynSentStartTime = 0;
    }

    ~SeastarClient() {}

    seastar::future<int> ConnectToOne(int64_t connId, std::string serverAddr, bool isReconnect = false);

    void Start(int64_t connId, seastar::connected_socket&& sk, const std::string& serverAddr);

    ClientNetworkContext* GetClientNetworkContext();

    seastar::future<> Connect(std::string serverAddr, int64_t connId, int64_t userThreadId
        , uint64_t seq, bool isReconnect = false);

    // the func will be called by seastar-core when finished
    // the func name must be 'stop', you can NOT use 'Stop'
    seastar::future<> stop()
    {
        return seastar::make_ready_future();
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

    uint64_t GetTimeoutInterval() const
    {
        return mTimeoutInterval;
    }
private:
    int64_t mConnConnected;
    ClientNetworkContext* mClientNetworkContext;
    // whether enable tcp_no_delay flag in socket(connection)
    bool mTcpNoDelayOn;
    int mTcpKeepAliveIdle;
    int mTcpKeepAliveCnt;
    int mTcpKeepAliveInterval;

    // for statstics
    SystemStats mSystemStats;

    // client send SYN to server, and change the status to SYN_SENT,
    // then will be received the ACK from the server,
    // mSynSentStartTime is the timestamp of client send SYN to server.
    uint64_t mSynSentStartTime;

    // packet timeout interval
    uint64_t mTimeoutInterval;
};

} // namespace network
} // namespace ps

#endif /* SERVICE_CLIENT_HH_ */

#ifndef ML_PS5_NET_CONNECT_H
#define ML_PS5_NET_CONNECT_H

#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "core/ps_coding/unittest/echo_request_serializer.hh"
#include "service/session_context.hh"
#include "service/client_network_context.hh"
#include "service/client.hh"

namespace ps
{
namespace network
{

class ConnectRequestItem : public ps::network::SeastarWorkItem
{
public:
    ConnectRequestItem(ps::network::NetworkContext* networkContext, int64_t userThreadId,
            unsigned cpuId, int64_t serverId, std::string serverAddr, bool reconnect)
        : ps::network::SeastarWorkItem(networkContext, serverId), mUserThreadId(userThreadId)
        , mCpuId(cpuId), mServerAddr(serverAddr), mReconnect(reconnect)
    {}

    seastar::future<> Run() override
    {
        ClientNetworkContext* networkContext = dynamic_cast<ClientNetworkContext *>(GetNetworkContext());
        if (networkContext == NULL) assert("Bad client network context.");
        // connect or reconnect to specific server
        return networkContext->mShardClients.invoke_on(mCpuId, &SeastarClient::Connect,
            mServerAddr, GetRemoteId(), mUserThreadId, GetSequence(), mReconnect).then([] ()
        {
            return seastar::make_ready_future<>();
        });
    }

private:
    int64_t mUserThreadId;
    unsigned mCpuId;
    std::string mServerAddr;
    bool mReconnect;
};

} // namespace net
} // ps

#endif // ML_PS5_NET_CONNECT_H


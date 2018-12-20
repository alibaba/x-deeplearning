#ifndef SEASTAR_SESSION_CONTEXT_HH_
#define SEASTAR_SESSION_CONTEXT_HH_

#include "core/future.hh"
#include "core/future-util.hh"
#include "service/base_connection.hh"
#include "service/seastar_status.hh"
#include "core/ps_coding/message_serializer.hh"

#include <iostream>

namespace ps
{
namespace network
{

class NetworkContext;

class SeastarSessionContext : public SessionContext
{
public:
    SeastarSessionContext(BaseConnection* conn, NetworkContext* ctx) 
    : SessionContext(conn), mNetworkContext(ctx) {}

    seastar::future<> Write(ps::coding::MessageSerializer* serializer);

    seastar::future<> Reconnect(std::string serverAddr, int64_t remoteId, int64_t userThreadId, uint64_t sequenceId);

    ~SeastarSessionContext();

    void ReportError(int64_t serverId, int64_t userThreadId, uint64_t seq, ps::network::ConnectionStatus status
        , int64_t eno);

private:
    seastar::future<> DoWrite(ps::coding::MessageSerializer* serializer);
    NetworkContext* mNetworkContext;
};

} // namespace network
} // namespace ps

#endif /*SEASTAR_SESSION_CONTEXT_HH_*/

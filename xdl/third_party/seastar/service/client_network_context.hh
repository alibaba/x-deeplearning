#ifndef PS_NETWORK_CLIENT_NETWORK_CONTEXT_HH_
#define PS_NETWORK_CLIENT_NETWORK_CONTEXT_HH_

#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_common.hh"
#include "service/network_context.hh"
#include "core/distributed.hh"
#include "service/client.hh"
#include "stats_printer.hh"
#include "service/queue_poller.hh"
#include <string>
#include <stdexcept>
#include <iostream>

namespace ps
{
namespace network
{
using namespace std;

class ClientNetworkContext : public ps::network::NetworkContext
{
public:
    ClientNetworkContext(int32_t serverCount, int32_t clientCount, int32_t coreCount, int32_t userThreadCount,
            string queueName = "SEASTAR") :
        NetworkContext(serverCount, clientCount, coreCount, userThreadCount, queueName)
    {
        // In fact: u can set the mServerCount as the max available servers count,
        // that you can dynamic add servers(connect to new servers) if you want.
        // The most important work: u must guarantee the right server id
        mSessionsContext.resize(mServerCount);
        mStatsPrinter = new ClientStatsPrinter(mShardClients);
    }

    ~ClientNetworkContext()
    {
        if (mStatsPrinter)
        {
            delete mStatsPrinter;
            mStatsPrinter = NULL;
        }
    }

    void SetSessionContextOfId(std::unique_ptr<SessionContext>&& sc, int64_t serverId)
    {
        mSessionsContext[serverId] = std::move(sc);
    }

    SessionContext* GetSessionOfId(int64_t serverId) const override
    {
        if (serverId < 0 || serverId >= mServerCount) return NULL;
        return mSessionsContext[serverId].get();
    }

    std::unique_ptr<SessionContext>* GetSessionAddrOfId(int64_t serverId)
    {
        if (serverId < 0 || serverId >= mServerCount) return NULL;
        return &(mSessionsContext[serverId]);
    }

    unsigned GetCoreIdOfServerId(int64_t serverId) const
    {
        if (serverId < 0 || serverId >= mServerCount)
        {
            throw std::runtime_error("Invalid server id: " + std::to_string(serverId));
        }
        unsigned targetId = GetCoreByServerId(serverId, mCoreCount);
        if (seastar::smp::seperate_server_client)
        {
            // nothing now
        }
        return targetId;
    }

    unsigned GetQueueIdOfServerId(int64_t serverId) const
    {
        if (serverId < 0 || serverId >= mServerCount)
        {
            throw std::runtime_error("Get queue id, invalid server id: " + std::to_string(serverId));
        }
        return GetCoreByServerId(serverId, mCoreCount);
    }

    static int32_t GetCoreByServerId(int32_t serverId, int32_t coreCount)
    {
        return serverId % coreCount;
    }

    seastar::future<int> DoConnectToAllServers(unsigned cur,
            std::vector<std::tuple<int64_t, string>> serverAddrs)
    {
        return mShardClients.invoke_on(GetCoreIdOfServerId(std::get<0>(serverAddrs[cur])),
            &SeastarClient::ConnectToOne,
            std::get<0>(serverAddrs[cur]),
            std::get<1>(serverAddrs[cur]), false).then([this, cur, serverAddrs] (auto code)
        {
            if (code == -1)
            {
                std::cout << "Connect to " << std::get<1>(serverAddrs[cur]) << " failed." << std::endl;
                return seastar::make_ready_future<int>(-1);
            }

            if (cur + 1 == serverAddrs.size())
            {
                //std::cout << "Connect to all servers successfully." << std::endl;
                return seastar::make_ready_future<int>(0);
            }

            //std::cout << "Connect to " << std::get<1>(serverAddrs[cur]) << " successfully." << std::endl;
            return this->DoConnectToAllServers(cur + 1, serverAddrs);
        });
    }

    seastar::future<int> ConnectToAllServers(const std::vector<std::tuple<int64_t, string> >& serverAddrs)
    {
        if (serverAddrs.size() == 0)
        {
            return seastar::make_ready_future<int>(0);
        }

        return DoConnectToAllServers(0, serverAddrs);
    }

    ps::network::Future<ps::network::Item> ReconnectToServer(ps::network::Item* item, int src,
        int dest, ps::service::seastar::Closure* cs)
    {
        return mQueueHubPair.first->Enqueue(item, src, dest, cs);
    }

    ps::network::Future<ps::network::Item> ConnectToServer(ps::network::Item* item, int src,
        int dest, ps::service::seastar::Closure* cs)
    {
        return mQueueHubPair.first->Enqueue(item, src, dest, cs);
    }

    int CreateClientInstance(int ac, char** av,
            const std::vector<std::tuple<int64_t, string> >& serverAddrs, uint64_t timeoutInterval)
    {
        mApp.add_options()
            ("tcp_nodelay_on", boost::program_options::value<bool>()->default_value(false),
             "Enable tcp nodelay to cacel Nagle algorithm.")
            ("tcp_keep_alive_idle", boost::program_options::value<int>()->default_value(0),
             "Enable tcp keep alive time.")
            ("tcp_keep_alive_cnt", boost::program_options::value<int>()->default_value(6),
             "Enable tcp keep alive cnt.")
            ("tcp_keep_alive_interval", boost::program_options::value<int>()->default_value(10),
             "Enable tcp keep alive interval.");

        return mApp.run_deprecated(ac, av, [&]
        {
            auto&& config = mApp.configuration();
            auto tcpNoDelayOn = config["tcp_nodelay_on"].as<bool>();
            auto tcpKeepAliveIdle = config["tcp_keep_alive_idle"].as<int>();
            auto tcpKeepAliveCnt = config["tcp_keep_alive_cnt"].as<int>();
            auto tcpKeepAliveInterval = config["tcp_keep_alive_interval"].as<int>();

            seastar::engine().at_exit([&]
            {
                return mShardClients.stop();
            });

            return mShardClients.start(this, tcpNoDelayOn, tcpKeepAliveIdle, tcpKeepAliveCnt, tcpKeepAliveInterval, timeoutInterval).then([this, serverAddrs]
            {
                if (serverAddrs.size() == 0)
                {
                    return seastar::make_ready_future<int>(0);
                }

                return ConnectToAllServers(serverAddrs).then([] (auto code)
                {
                    // connect to some servers failed
                    if (code == -1)
                    {
                        // exit seastar
                        engine().exit(-1);
                        return seastar::make_ready_future<int>(-1);
                    }
                    return seastar::make_ready_future<int>(0);
                });
            }).then([&] (auto code)
            {
                if (code == 0)
                {
                    // delate the stats printer in release version
                    //this->GetStatsPrinter()->Start();
                    this->SetRunning(true);

                    //std::cout << "Client start ... " << std::endl;
                }
                else
                {
                    std::cout << "Client can't be launched." << std::endl;
                }

                WaitLock::GetInstance().Release();
            });
        });
    }

    // @hasResponse: indicate wether we hope a response for the request
    ps::network::Future<ps::network::Item> EnqueueItem(ps::network::Item* item, int src, unsigned dest,
        ps::service::seastar::Closure* cs, bool hasResponse = true)
    {
        return mQueueHubPair.first->Enqueue(item, src, dest, cs, hasResponse);
    }
    ps::network::Future<ps::network::Item> EnqueueItem(ps::network::Item* item, unsigned dest,
        ps::service::seastar::Closure* cs, bool hasResponse = true)
    {
        int32_t src = ps::network::QueueHub<ps::network::Item>::GetPortNumberOnThread();
        return mQueueHubPair.first->Enqueue(item, src, dest, cs, hasResponse);
    }

    void PushItemBack(ps::network::Item* item, int src, int dest)
    {
        mQueueHubPair.second->EnqueueBack(item, src, dest);
    }

    ClientStatsPrinter* GetStatsPrinter() const
    {
        return mStatsPrinter;
    }

    // set the response processer id for a specific connection(server)
    // connId == serverId
    void SetResponseProcessorIdOfServer(int64_t serverId, uint64_t id)
    {
        mSessionsContext[serverId]->SetProcessorId(id);
    }

    uint64_t GetResponseProcessorIdOfServer(int64_t serverId)
    {
        return mSessionsContext[serverId]->GetProcessorId();
    }

    // stop: It should be a sync function
    void Stop(ps::network::Item* item, int src, int dest, ps::service::seastar::Closure* cs)
    {
        mQueueHubPair.first->Enqueue(item, src, dest, cs).Get();
        SetRunning(false);
    }

    // timeoutInterval: packet timeout interval in ms
    void operator() (int ac, char**args,
        const std::vector<std::tuple<int64_t, string> >& serverAddrs,
        uint64_t timeoutInterval = 1000)
    {
        int ret = CreateClientInstance(ac, args, serverAddrs, timeoutInterval);
        std::cout << "Seastar exit code = " << ret << std::endl;
    }

    void operator() (int ac, char**args, uint64_t timeoutInterval = 1000)
    {
        int ret = CreateClientInstance(ac, args, std::vector<std::tuple<int64_t, string> >(), timeoutInterval);
        std::cout << "Seastar exit code = " << ret << std::endl;
    }

    seastar::future<> Stop()
    {
        return mShardClients.stop();
    }

private:
    ClientStatsPrinter* mStatsPrinter;

public:
    seastar::distributed<SeastarClient> mShardClients;
};

} // namespace network
} // namespace ps

#endif // PS_NETWORK_CLIENT_NETWORK_CONTEXT_HH_

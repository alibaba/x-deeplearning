#ifndef SERVER_NETWORK_CONTEXT_HH_
#define SERVER_NETWORK_CONTEXT_HH_

#include "service/network_context.hh"
#include "core/distributed.hh"
#include "core/app-template.hh"
#include "service/server.hh"
#include <iostream>
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/queue_hub_future.hh"
#include "core/ps_queue_hub/queue_hub.hh"
#include "stats_printer.hh"

namespace bpo = boost::program_options;

namespace ps
{
namespace network
{

class ServerNetworkContext : public ps::network::NetworkContext
{
public:
    ServerNetworkContext(int32_t serverCount, int32_t clientCount, int32_t coreCount, int32_t uThreadCount, string queueName = "SEASTAR") :
        NetworkContext(serverCount, clientCount, coreCount, uThreadCount, queueName)
    {
        mStatsPrinter = new ServerStatsPrinter(mShardServers);
    }

    virtual ~ServerNetworkContext()
    {
        if (mStatsPrinter) delete mStatsPrinter;
    }

    int CreateClientInstance(int ac, char** av)
    {
        mApp.add_options()
            ("port", bpo::value<uint16_t>()->default_value(10000), "Port listen on")
            ("tcp_nodelay_on", boost::program_options::value<bool>()->default_value(false), "Enable tcp nodelay to cacel Nagle algorithm.")
            ("tcp_keep_alive_idle", boost::program_options::value<int>()->default_value(0),
             "Enable tcp keep alive time.")
            ("tcp_keep_alive_cnt", boost::program_options::value<int>()->default_value(6),
             "Enable tcp keep alive cnt.")
            ("tcp_keep_alive_interval", boost::program_options::value<int>()->default_value(10),
             "Enable tcp keep alive interval.");

        return mApp.run_deprecated(ac, av, [&]
        {
            seastar::engine().at_exit([&]
            {
                return mShardServers.stop();
            });

            auto&& config = mApp.configuration();
            uint16_t port = config["port"].as<uint16_t>();
            bool tcpNoDelayOn = config["tcp_nodelay_on"].as<bool>();
            auto tcpKeepAliveIdle = config["tcp_keep_alive_idle"].as<int>();
            auto tcpKeepAliveCnt = config["tcp_keep_alive_cnt"].as<int>();
            auto tcpKeepAliveInterval = config["tcp_keep_alive_interval"].as<int>();

            return mShardServers.start(port, tcpNoDelayOn, tcpKeepAliveIdle, tcpKeepAliveCnt, tcpKeepAliveInterval).then([&]
            {
                return mShardServers.invoke_on_all(&SeastarServer::Start);
            }).then([&, port]
            {
                std::cout << "Seastar server listen on: " << port << "\n";
            });
        });
    }

    seastar::future<> InvokeOnSomeCores(unsigned start, unsigned end)
    {
	return InvokeOn(start, end);
    }

    // start server at cores, from cur to max
    seastar::future<> InvokeOn(unsigned cur, unsigned max)
    {
	if (cur > max) return seastar::make_ready_future<>();
        return mShardServers.invoke_on(cur, &SeastarServer::Start).then([this, cur, max] () {
	    std::cout << "server start at cpu: " << cur << std::endl;
	    return this->InvokeOn(cur+1, max);
	});
    }

    ServerStatsPrinter* GetStatsPrinter() const
    {
        return mStatsPrinter;
    }

    void operator() (int ac, char**args)
    {
        CreateClientInstance(ac, args);
    }

    seastar::future<> Stop()
    {
        return mShardServers.stop();
    }
private:
    ServerStatsPrinter* mStatsPrinter;

public:
    seastar::distributed<SeastarServer> mShardServers;
};

} // namespace network
} // namespace ps

#endif /*SERVER_NETWORK_CONTEXT_HH_*/

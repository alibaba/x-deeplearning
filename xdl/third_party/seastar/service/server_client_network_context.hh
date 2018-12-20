#ifndef SERVER_CLIENT_NETWORK_CONTEXT_HH_
#define SERVER_CLIENT_NETWORK_CONTEXT_HH_

#include "core/app-template.hh"
#include "core/ps_common.hh"
#include "service/client_network_context.hh"
#include "service/server_network_context.hh"

#include <iostream>

namespace ps
{
namespace network
{

class ServerClientNetworkContext
{
public:
    // NOT in seperate mode
    ServerClientNetworkContext(int32_t serverCount, int32_t clientCount, int32_t coreCount
        , int32_t clientThreadCount, int32_t serverThreadCount)
    : mClientNetworkContext(serverCount, clientCount, coreCount, clientThreadCount, "SEASTAR")
    , mServerNetworkContext(serverCount, clientCount, coreCount, serverThreadCount, "SEASTAR2")
    , mServerCoresCount(coreCount), mClientCoresCount(coreCount)
    {
        std::cout << "server_client mode: " << serverCount << ", " << clientCount << ", " << coreCount
	    << ", " << clientThreadCount << ", " << serverThreadCount << endl;
    }

    // In seperate server_client mode
    // you should offer serverCoreCount and clientCoreCount
    // we use them to create MxN queue.
    ServerClientNetworkContext(int32_t serverCount, int32_t clientCount, int32_t serverCoreCount
        , int32_t clientCoreCount, int32_t clientThreadCount, int32_t serverThreadCount)
    : mClientNetworkContext(serverCount, clientCount, clientCoreCount, clientThreadCount, "SEASTAR")
    , mServerNetworkContext(serverCount, clientCount, serverCoreCount, serverThreadCount, "SEASTAR2")
    , mServerCoresCount(serverCoreCount), mClientCoresCount(clientCoreCount)
    {
        std::cout << "server_client mode: " << serverCount << ", " << clientCount << ", " << serverCoreCount
	    << ", " << clientCoreCount << ", " << clientThreadCount << ", " << serverThreadCount << endl;
    }

    // @seperateCore: if true, client and server will not run in the same core.
    void operator() (int ac, char**args, uint64_t timeIntervel = 1000, bool seperateCore = false)
    {
        int ret = CreateInstance(ac, args, std::vector<std::tuple<int64_t, string> >(), timeIntervel, seperateCore);
        std::cout << "Seastar exit code = " << ret << std::endl;
    }

    void operator() (int ac, char**args,
        const std::vector<std::tuple<int64_t, string> >& serverAddrs, uint64_t timeIntervel = 1000, bool seperateCore = false)
    {
        int ret = CreateInstance(ac, args, serverAddrs, timeIntervel, seperateCore);
        std::cout << "Seastar exit code = " << ret << std::endl;
    }

    ClientNetworkContext& GetClient()
    {
        return mClientNetworkContext;
    }

    ServerNetworkContext& GetServer()
    {
        return mServerNetworkContext;
    }

    // stop: It should be a sync function
    void Stop(ps::network::Item* item, int src, int dest, ps::service::seastar::Closure* cs)
    {
        mClientNetworkContext.Stop(item, src, dest, cs);
    }

private:
    int CreateInstance(int ac, char** av,
       	const std::vector<std::tuple<int64_t, string> >& serverAddrs
   	, uint64_t timeIntervel, bool seperateCore)
    {
        mApp.add_options()
            // default add server_client_mode here
            ("server_client_mode", bpo::value<bool>()->default_value(true), "wether start server and client at one node")
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
            auto&& config = mApp.configuration();
            auto port = config["port"].as<uint16_t>();
            auto tcpNoDelayOn = config["tcp_nodelay_on"].as<bool>();
            auto tcpKeepAliveIdle = config["tcp_keep_alive_idle"].as<int>();
            auto tcpKeepAliveCnt = config["tcp_keep_alive_cnt"].as<int>();
            auto tcpKeepAliveInterval = config["tcp_keep_alive_interval"].as<int>();

            seastar::engine().at_exit([&]
            {
                return this->GetServer().Stop().then([this] () {
                    return this->GetClient().Stop();
                });
            });
            // start client
            return this->GetClient().mShardClients.start(&(this->GetClient()), tcpNoDelayOn, tcpKeepAliveIdle, tcpKeepAliveCnt, tcpKeepAliveInterval, timeIntervel).then([this, port, serverAddrs]
            {
                if (serverAddrs.size() == 0)
                {
                    return seastar::make_ready_future<int>(0);
                }

                return this->GetClient().ConnectToAllServers(serverAddrs).then([] (auto code)
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
            }).then([this, port, tcpNoDelayOn, tcpKeepAliveIdle, tcpKeepAliveCnt, tcpKeepAliveInterval, seperateCore] (auto code)
            {
                if (code == 0)
                {
                    // disable the stats printer in release version
                    //this->GetClient().GetStatsPrinter()->Start();

                    //std::cout << "Client start ... " << std::endl;

                    // start server
                    return this->GetServer().mShardServers.start(port, tcpNoDelayOn, tcpKeepAliveIdle, tcpKeepAliveCnt, tcpKeepAliveInterval).then([&]
                    {
                        // In seperate mode, server can not start at all cores
                        // server cpu id start from 0 to mServerCoresCount-1
                        // client cpu id start from mServerCoresCount to count-1
                        if (seperateCore)
                        {
                            // server cores range: mClientCoresCount ~ mClientCoresCount + mServerCoresCount - 1
                            return this->GetServer().InvokeOnSomeCores(mClientCoresCount, mClientCoresCount + mServerCoresCount - 1);
                        }
                        else
                        {
                            return this->GetServer().mShardServers.invoke_on_all(&SeastarServer::Start);
                        }
                    }).then([&, port]
                    {
                        //std::cout << "Seastar server listen on: " << port << std::endl;
                        this->GetClient().SetRunning(true);

                        WaitLock::GetInstance().Release();
                        return seastar::make_ready_future<>();
                    });                    
                }
                else
                {
                    std::cout << "Client can't be launched." << std::endl;

                    WaitLock::GetInstance().Release();
                    return seastar::make_ready_future<>();
                }
            });
        });
    }

private:
    seastar::app_template mApp;
    ClientNetworkContext mClientNetworkContext;
    ServerNetworkContext mServerNetworkContext;
    int32_t mServerCoresCount;
    int32_t mClientCoresCount;
};

} // namespace network
} // namespace ps

#endif /* SERVER_CLIENT_NETWORK_CONTEXT_HH_  */

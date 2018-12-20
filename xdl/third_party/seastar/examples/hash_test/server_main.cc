#include "common/LogFacade.h"
#include "common/option_parser.h"
#include "server/server_service.h"
#include "scheduler/scheduler_impl.h"
#include "scheduler/placementer.h"
#include <thread>

// server run
int ServerRun(int argc, char** argv)
{
    ps::OptionParser optParser;
    optParser.addOption("-sp", "--scheduler_kv_path", "scheduler_kv_path", ps::OptionParser::OPT_STRING, true);
    optParser.addOption("-p", "--port", "port", ps::OptionParser::OPT_INT32, true);
    optParser.addOption("-si", "--server_id", "server_id", ps::OptionParser::OPT_INT32, true);
    if (!optParser.parseArgs(argc, argv))
    {
        LOG_ERROR("Parse Server Args Error");
        return -1;
    }

    std::string schedulerKvPath;
    int port;
    int serverId;

    optParser.getOptionValue("scheduler_kv_path", schedulerKvPath);
    optParser.getOptionValue("port", port);
    optParser.getOptionValue("server_id", serverId);

    ps::server::ServerService service(schedulerKvPath, port, serverId);
    service.Init();
  
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3600));
    }
 
    return 0;
}


// scheduler run
int SchedulerRun(int argc, char** argv)
{
    ps::OptionParser optParser;
    optParser.addOption("-sp", "--scheduler_kv_path", "scheduler_kv_path", ps::OptionParser::OPT_STRING, true);
    optParser.addOption("-cp", "--checkpoint_path", "checkpoint_path", ps::OptionParser::OPT_STRING, true);
    optParser.addOption("-p", "--port", "port", ps::OptionParser::OPT_INT32, true);
    optParser.addOption("-sn", "--server_num", "server_num", ps::OptionParser::OPT_INT32, true);
    optParser.addOption("-snet", "--server_network_limit", "server_network_limit", ps::OptionParser::OPT_INT32, true); //GB/s
    optParser.addOption("-smem", "--server_memory_limit", "server_memory_limit", ps::OptionParser::OPT_INT32, true); //GB
    optParser.addOption("-sqps", "--server_query_limit", "server_query_limit", ps::OptionParser::OPT_INT32, true);

    if (!optParser.parseArgs(argc, argv))
    {
        LOG_ERROR("argument error");
        return -1;
    }

    std::string schedulerKvPath;
    std::string checkpointPath;
    int port;
    int serverNum;
    int serverNetworkLimit;
    int serverMemoryLimit;
    int serverQueryLimit;

    optParser.getOptionValue("scheduler_kv_path", schedulerKvPath);
    optParser.getOptionValue("checkpoint_path", checkpointPath);
    optParser.getOptionValue("port", port);
    optParser.getOptionValue("server_num", serverNum);
    optParser.getOptionValue("server_network_limit", serverNetworkLimit);
    optParser.getOptionValue("server_memory_limit", serverMemoryLimit);
    optParser.getOptionValue("server_query_limit", serverQueryLimit);

    ps::scheduler::Placementer::Arg placementArg
    {
        .net = (size_t)serverNetworkLimit * (1 << 30),
        .mem = (size_t)serverMemoryLimit * (1 << 30),
        .query = (size_t)serverQueryLimit
    };

    ps::scheduler::SchedulerImpl service(serverNum, schedulerKvPath, port, checkpointPath, placementArg);
    service.Start();

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(3600));
    }

    return 0;
}

int main(int argc, char** argv)
{
    ps::OptionParser optParser;
    optParser.addOption("-r", "--role", "role", ps::OptionParser::OPT_STRING, true);
    std::string role;

    if (!optParser.parseArgs(argc, argv))
    {
        LOG_ERROR("Must Specify role");
        return -1;
    }

    optParser.getOptionValue("role", role);

    // if start scheduler
    if (role == "scheduler")
    {
        return SchedulerRun(argc, argv);
    }
    // if start server
    else if (role == "server")
    {
        return ServerRun(argc, argv);
    }
    // error
    else
    {
        LOG_ERROR("Role must be scheduler or server");
        return -1;
    }
}



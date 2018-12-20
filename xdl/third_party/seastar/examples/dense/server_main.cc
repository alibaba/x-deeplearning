#include <iostream>
#include <string.h>
#include <boost/thread.hpp>
#include "service/server_network_context.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "core/ps_queue_hub/queue_hub.hh"

using namespace std;

vector<int64_t> gStatVec;
vector<int64_t> gMaxVec;
vector<int64_t> gMinVec;

void Hello(int32_t id)
{
    pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr
        = ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("SEASTAR");
    ps::network::QueueHub<ps::network::Item>* outputHub = qhr.second;
    ps::network::Future<ps::network::Item> myFuture = outputHub->GetFuture(0, id);

    while(true)
    {
        myFuture.Get();
    }
}

int main(int ac, char** as)
{
    const int32_t serverNum = 1;
    const int32_t clientNum = 1;
    const int32_t coreNum = 1;
    const int32_t userThreadNum = 2;
    ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("SEASTAR", userThreadNum, coreNum);
    boost::thread serverComputeThread0(&Hello, 0);
    //boost::thread serverComputeThread1(&Hello, 1);

    ps::network::ServerNetworkContext snc(serverNum, clientNum, coreNum, userThreadNum);

    char** args;
    args = new char*[3];
    args[0] = new char[100];
    args[1] = new char[100];
    args[2] = new char[100];

    strcpy(args[0], as[1]);
    //strcpy(args[0], "--smp=1");
    strcpy(args[1], as[2]);
    //strcpy(args[1], "--cpuset=14");
    strcpy(args[2], as[3]);
    //strcpy(args[2], "--port=10001");

    // 线程中启动server
    boost::thread sncThread(boost::ref(snc), 3, args);

    sncThread.join();
    serverComputeThread0.join();
    //serverComputeThread1.join();

    return 0;
}


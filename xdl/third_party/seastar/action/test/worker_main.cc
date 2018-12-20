#include <iostream>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <tuple>
#include <vector>
#include <limits>
#include <boost/thread.hpp>
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/queue_hub.hh"
#include "core/ps_queue_hub/queue_hub_future.hh"
#include "service/client_network_context.hh"
#include "service/network_context_helper.hh"
#include "action/utility.hh"
#include "action/push_action.hh"

using namespace std;

void Hello(int32_t myId, int32_t coreNum, int32_t testTimes)
{
    pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr
        = ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("SEASTAR");
    ps::network::QueueHub<ps::network::Item>* outputHub = qhr.second;
    const int32_t total = 2 * testTimes;
    int32_t counter = 0;
    while(true)
    {
        for (int32_t i = 0; i < coreNum; ++i)
        {
            ps::network::Item* ptr = NULL;
            ptr = outputHub->Dequeue(myId, i);
            if (ptr != NULL)
            {
                ++counter;
                if (ptr->GetItemType() == ps::network::PsEngine)
                {
                    ps::network::PsWorkItem* psItem = static_cast<ps::network::PsWorkItem*>(ptr);
                    psItem->Run();
                }
                if (counter == total)
                {
                    return;
                }
            }
        }
    }
}

int main(int ac, char* as[])
{
    //gStatVec.resize(14, 0);
    //gMaxVec.resize(14, numeric_limits<int64_t>::min());
    //gMinVec.resize(14, numeric_limits<int64_t>::max());

    const int32_t clientNum = 1;
    const int32_t userThreadNum = 1;
    int32_t coreNum = 2;
    char** args = new char*[3];
    args[0] = new char[100];
    args[1] = new char[100];
    args[2] = new char[100];
    if (ac > 5)
    {
        cout << "error input!" << endl;
        return 0;
    }
    int32_t serverNum = 0;
    int32_t testTimes = 0;
    if (ac == 1)
    {
        testTimes = 1;
        serverNum = 1;
        strcpy(args[0], "--smp=1");
        strcpy(args[1], "--cpuset=12");
        coreNum = 1;
    }
    else if (ac == 2)
    {
        testTimes = atoi(as[1]);
        serverNum = 1;
        strcpy(args[0], "--smp=1");
        strcpy(args[1], "--cpuset=12");
        coreNum = 1;
    }
    else if (ac == 3)
    {
        testTimes = atoi(as[1]);
        serverNum = atoi(as[2]);
        strcpy(args[0], "--smp=1");
        strcpy(args[1], "--cpuset=12");
        coreNum = 1;
    }
    else if (ac == 5)
    {
        testTimes = atoi(as[1]);
        serverNum = atoi(as[2]);
        strcpy(args[0], as[3]);
        strcpy(args[1], as[4]);
        const int32_t len = strlen(args[0]);
        coreNum = atoi(args[0] + len - 1);
    }
    else
    {
        testTimes = 1;
        serverNum = 1;
        strcpy(args[0], "--smp=1");
        strcpy(args[1], "--cpuset=12");
        coreNum = 1;
    }
    cout << "coreNum=" << coreNum << endl;
    cout << "serverNum=" << serverNum << endl;
    cout << "testTimes=" << testTimes << endl;
    ps::network::gServerCount = serverNum;
    ps::network::gCoreCount = coreNum;
    ps::network::gUserCount = userThreadNum;
    ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("SEASTAR", userThreadNum, coreNum);
    //const int32_t currentThreadId = 0;
    //boost::thread workerDequeueThread0(&Hello, currentThreadId, coreNum, testTimes);

    vector<std::tuple<int64_t, string> > serverAddrsBak;
    serverAddrsBak.push_back(std::make_tuple(0, "127.0.0.1:17000"));
    serverAddrsBak.push_back(std::make_tuple(1, "127.0.0.1:17001"));
    serverAddrsBak.push_back(std::make_tuple(2, "127.0.0.1:17002"));
    serverAddrsBak.push_back(std::make_tuple(3, "127.0.0.1:17003"));

    //serverAddrsBak.push_back(std::make_tuple(0, "10.101.200.223:17000"));
    //serverAddrsBak.push_back(std::make_tuple(10, "10.101.200.223:17010"));
    //serverAddrsBak.push_back(std::make_tuple(20, "10.101.200.223:17020"));

    vector<std::tuple<int64_t, string> > serverAddrs;
    for (int32_t i = 0; i < serverNum; ++i)
    {
        serverAddrs.push_back(serverAddrsBak[i]);
    }

    ps::network::ClientNetworkContext& cnc = ps::network::NetworkContextHelper::GetNetworkContext();
    boost::thread cncThread(boost::ref(cnc), 2, args, serverAddrs);

    sleep(1);
    const int32_t doubleCount = 64;
    vector<double> source(doubleCount, 1.0);
    int32_t shardId = 0;
    ps::action::PushAction<double> act(source, 0, doubleCount);
    act.Run();

    /*
    cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Send End$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;
    */

    //workerDequeueThread0.join();
    cncThread.join();

    return 0;
}


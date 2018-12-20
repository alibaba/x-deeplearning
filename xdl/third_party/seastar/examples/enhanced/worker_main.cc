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
#include <boost/thread.hpp>
#include "service/client_network_context.hh"
#include "examples/dense/echo_request_item.hh"
#include "examples/dense/worker_response_item.hh"

using namespace std;

vector<int64_t> gStatVec;
vector<int64_t> gMaxVec;
vector<int64_t> gMinVec;

uint64_t CoutCurrentTimeOnNanoSec()
{
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    uint64_t current = (uint64_t)start_time.tv_sec * 1000000000UL + start_time.tv_nsec;
    //cout << "current=" << current << endl;
    return current;
}

void Hello(int32_t myId, int32_t coreNum, int32_t testTimes)
{
    pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr
        = ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("SEASTAR");
    ps::network::QueueHub<ps::network::Item>* outputHub = qhr.second;
    //ps::network::Future<ps::network::Item> myFuture = outputHub->GetFuture(0, id);
    //while(true)
    //{
        //myFuture.Get();
    //}
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
    gStatVec.resize(14, 0);
    gMaxVec.resize(14, numeric_limits<int64_t>::min());
    gMinVec.resize(14, numeric_limits<int64_t>::max());

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

    ps::network::ClientNetworkContext cnc(serverNum, clientNum/*non-sense now*/, coreNum, userThreadNum);
    boost::thread cncThread(boost::ref(cnc), 2, args, serverAddrs);

    sleep(1);
    cout << "......................." << endl;
    vector<ps::network::Future<ps::network::Item>> fv0;
    vector<ps::network::Future<ps::network::Item>> fv1;
    vector<vector<ps::network::Future<ps::network::Item>>> fuMatrix(serverNum);
    //const int32_t doubleCount = 1024 * 128;
    const int32_t doubleCount = 32;
    vector<double> source(doubleCount, 1.0);
    int32_t shardId = 0;

    cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Prepare Begin$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;
    cout << "ps::coding::unittest::WorkerResponseItem<double>::kPreTimes="
        << ps::coding::unittest::WorkerResponseItem<double>::kPreTimes << endl;
    for (int32_t i = 0; i < ps::coding::unittest::WorkerResponseItem<double>::kPreTimes; ++i)
    {
		int32_t j = i % serverNum;
		ps::coding::unittest::EchoRequestItem<double>* pItem
			= new ps::coding::unittest::EchoRequestItem<double>(
                    ps::network::QueueHub<ps::network::Item>::GetPortNumberOnThread(),
					//currentThreadId,
					&cnc, j,
					source, 0, doubleCount / 2, shardId);
		ps::network::Future<ps::network::Item> f
			= cnc.EnqueueItem(pItem,
					//ps::network::QueueHub<ps::network::Item>::GetPortNumberOnThread(),
					//currentThreadId,
					cnc.GetCoreIdOfServerId(j), NULL);
        f.Get();
    }
    cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Prepare End$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;
    cout << "$$$gStatVec=" << gStatVec.back() << endl;
    cout << "$$$gMaxVec=" << gMaxVec.back() << endl;
    cout << "$$$gMinVec=" << gMinVec.back() << endl;
    cout << endl;
    sleep(1);

    uint64_t s0 = CoutCurrentTimeOnNanoSec();
    for (int32_t i = 0; i < testTimes; ++i)
    {
        //cout << "+++++++++++i=" << i << endl;
        int32_t j = i % serverNum;
        ps::coding::unittest::EchoRequestItem<double>* pItem
            = new ps::coding::unittest::EchoRequestItem<double>(
                    ps::network::QueueHub<ps::network::Item>::GetPortNumberOnThread(),
                    &cnc, j,
                    source, 0, doubleCount / 2, shardId);
        ps::network::Future<ps::network::Item> fu
            = cnc.EnqueueItem(pItem,
                    cnc.GetCoreIdOfServerId(j), NULL);
        fuMatrix[j].push_back(fu);
        //cout << "+++++++++++End+++i=" << i << endl;
    }
    cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Send End$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;
    for (int32_t i = 0; i < testTimes; ++i)
    {
        //cout << "+++++++++++i=" << i << endl;
        int32_t j = i % serverNum;
        int32_t k = i / serverNum;
        fuMatrix[j][k].Get();
        //cout << "+++++++++++End+++i=" << i << endl;
    }
    uint64_t s1 = CoutCurrentTimeOnNanoSec();
    cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Finished$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;
    sleep(1);

    cout << endl << "average time: " << endl;
    for (int32_t i = 0; i < gStatVec.size(); ++i)
    {
        cout << "___" << i << ":" << gStatVec[i] / testTimes << "  ";
    }

    cout << endl << "max time: " << endl;
    for (int32_t i = 0; i < gMaxVec.size(); ++i)
    {
        cout << "___" << i << ":" << gMaxVec[i] << "  ";
    }

    cout << endl << "min time: " << endl;
    for (int32_t i = 0; i < gMinVec.size(); ++i)
    {
        cout << "___" << i << ":" << gMinVec[i] << "  ";
    }
    cout << endl;

    cout << "gMaxVec.back()=" << gMaxVec.back() << endl;
    cout << "gMinVec.back()=" << gMinVec.back() << endl;
    cout << "gStatVec.back()=" << gStatVec.back() / testTimes << endl;
    cout << "Total Response Time: " << s1 - s0 << endl;

    //workerDequeueThread0.join();
    cncThread.join();

    return 0;
}


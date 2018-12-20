#include <iostream>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <tuple>
#include <vector>
#include <boost/thread.hpp>
#include "core/app-template.hh"
#include "core/distributed.hh"
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/seastar_work_item.hh"
#include "core/ps_queue_hub/queue_hub.hh"

using namespace std;

class TestItem : public ps::network::SeastarWorkItem
{
public:
    TestItem(int32_t instanceId) : ps::network::SeastarWorkItem(NULL, 0), mInstanceId(instanceId)
    {
    }
    seastar::future<> Run() override
    {
        return seastar::make_ready_future<>();
    }
    int32_t GetInstanceId() const { return mInstanceId; }
private:
    int32_t mInstanceId;
};

class SeastarMain
{
public:
    SeastarMain(
            pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr) :
        mQhr(qhr), mSendHub(NULL), mRecvHub(NULL), mTotalCount(0), mMyId(-1)
    {
    }
    ~SeastarMain()
    {}
    seastar::future<> Run(int32_t id, bool sender)
    {
        mMyId = id;
        if (sender)
        {
            // work as user thread
            mRecvHub = mQhr.second;
            mSendHub = mQhr.first;
        }
        else
        {
            // work as seastar core
            mRecvHub = mQhr.first;
            mSendHub = mQhr.second;
        }
        if (sender)
        {
            vector<ps::network::Future<ps::network::Item>>* fv
                = new vector<ps::network::Future<ps::network::Item>>;
            const int32_t targetCount = this->mSendHub->GetNCount();
            // work as user thread
            seastar::repeat([this, fv, targetCount]()
                    {
                        for (int32_t i = 0; i < targetCount; ++i)
                        {
                            TestItem* pItem0 = new TestItem(0);
                            ps::network::Future<ps::network::Item> f
                                = this->mSendHub->Enqueue(pItem0, mMyId, i);
                            fv->push_back(f);
                        }
                        return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::no);
                    });
        }
        else
        {
            // work as seastar core
            int32_t targetCount = mRecvHub->GetMCount();
            seastar::repeat([this, targetCount]()
                    {
                    //{
                    //struct timespec start_time;
                    //clock_gettime(CLOCK_MONOTONIC, &start_time);
                    //uint64_t current = (uint64_t)start_time.tv_sec * 1000000000UL + start_time.tv_nsec;
                    //cout << "current=" << current << endl;
                    //}
                    for (int32_t i = 0; i < targetCount; ++i)
                    {
                        TestItem* pItem0 = static_cast<TestItem*>(this->mRecvHub->Dequeue(i, mMyId));
                        if (pItem0 == NULL)
                        {
                            continue;
                        }
                        cout << "EnqueueTime:" << pItem0->mEnqueueTime
                        << " DequeueTime:" << pItem0->mDequeueTime
                        << " diff:" << pItem0->mDequeueTime - pItem0->mEnqueueTime
                        << endl;
                    }
                    return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::no);
                    });
        }
        return seastar::make_ready_future<>();
    }
    seastar::future<> stop()
    {
        return seastar::make_ready_future();
    }
private:
    pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> mQhr;
    ps::network::QueueHub<ps::network::Item>* mSendHub;
    ps::network::QueueHub<ps::network::Item>* mRecvHub;
    int32_t mTotalCount;
    int32_t mMyId;
};

int Hello(pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr,
        int ac, char** av)
{
    seastar::app_template app;
    seastar::distributed<SeastarMain>* shardMain = new seastar::distributed<SeastarMain>;
    return app.run_deprecated(ac, av, [&]
            {
            seastar::engine().at_exit([&]
                    {
                    return shardMain->stop();
                    });
            return shardMain->start(qhr).then([shardMain]
                    {
                    for (int32_t i = 0; i < 2; ++i)
                    {
                    const bool sender = (i == 1);
                    shardMain->invoke_on(i, &SeastarMain::Run, 0, sender);
                    }
                    });
            });
}

int main(int ac, char** av)
{
    const int32_t mc = 1;
    const int32_t nc = 1;
    pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*> qhr
        = ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("QUEUE_PERF_TEST", mc, nc);

    char** args = new char*[2];
    args[0] = new char[100];
    args[1] = new char[100];
    strcpy(args[0], "--smp=2");
    strcpy(args[1], "--cpuset=12,13");
    cout << "------------------" << endl;
    return Hello(qhr, 2, args);
}

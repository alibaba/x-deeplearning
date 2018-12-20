#include <iostream>
#include <string>
#include <memory>
//#include "network/boost/threadpool.hpp"
#include "ps_queue_hub/unittest/boost/threadpool.hpp"
#include "network/queue_hub.h"

using namespace std;
using namespace apsara::odps::ps;

static const int32_t kProducerCount = 8;
static const int32_t kConsumerCount = 6;
static const int64_t kRequestCount = 50000000;

static void Compute(int64_t x)
{
    ++x;
}

class Producer
{
public:
    Producer(int32_t id, network::QueueHub<function<void(int64_t)> >* qh) : mProducerId(id), mQueueHub(qh)
    {}
    void Produce()
    {
        for (int64_t i = 0; i < kRequestCount; ++i)
        {
            const int32_t consumerId = i % kConsumerCount;
            mQueueHub->SetA(bind(&Compute, i), mProducerId, consumerId);
        }
    }
private:
    int32_t mProducerId;
    network::QueueHub<function<void(int64_t)> >* mQueueHub;
};

class Consumer
{
public:
    Consumer(int32_t id, network::QueueHub<function<void(int64_t)> >* qh) : mConsumerId(id), mQueueHub(qh)
    {}
    void Consume()
    {
        while(true)
        {
            for (int32_t i = 0; i < kProducerCount; ++i)
            {
                function<void(int64_t)> f;
                if (mQueueHub->GetB(f, i, mConsumerId))
                {
                    f;
                }
            }
        }
    }
private:
    int32_t mConsumerId;
    network::QueueHub<function<void(int64_t)> >* mQueueHub;
};

class QueueHubUnitTestSuite
{
public:
    void Setup()
    {
    }

    void CleanUp()
    {
    }

    void QueueHubTestCase()
    {
        unique_ptr<network::QueueHub<function<void(int64_t)> > >
            qh(new network::QueueHub<function<void(int64_t)> >(kProducerCount, kConsumerCount));
        boost::threadpool::pool producerThreadPool(kProducerCount);
        boost::threadpool::pool consumerThreadPool(kConsumerCount);
        vector<unique_ptr<Consumer> > consumers(kConsumerCount);
        for (int32_t i = 0; i < kConsumerCount; ++i)
        {
            consumers[i].reset(new Consumer(i, qh.get()));
            consumerThreadPool.schedule(boost::bind(&Consumer::Consume, consumers[i].get()));
        }
        vector<unique_ptr<Producer> > producers(kProducerCount);
        for (int32_t i = 0; i < kProducerCount; ++i)
        {
            producers[i].reset(new Producer(i, qh.get()));
            producerThreadPool.schedule(boost::bind(&Producer::Produce, producers[i].get()));
        }
        consumerThreadPool.wait();
        producerThreadPool.wait();
    }
};

int main()
{
    QueueHubUnitTestSuite testSuite;
    testSuite.Setup();
    testSuite.QueueHubTestCase();
    testSuite.CleanUp();
    return 0;
}

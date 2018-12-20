#ifndef ML_PS5_NETWORK_PS_QUEUE_WORK_ACTION_H
#define ML_PS5_NETWORK_PS_QUEUE_WORK_ACTION_H

#include <iostream>
#include <string>
#include <memory>
#include <exception>
#include <vector>
#include <mutex>
#include "ps_work_item.hh"
#include "queue_hub_future.hh"
#include "queue_hub.hh"

namespace ps
{
namespace network
{
using namespace std;

template<typename T>
class PsWorkAction
{
public:
    PsWorkAction(int32_t seastarCoreCount) :
        mQueuePair(ps::network::QueueHubFactory::GetInstance().GetHub<T>("SEASTAR")),
        mSendQueue(mQueuePair.first),
        mRecvQueue(mQueuePair.second),
        kSeastarCoreCount(seastarCoreCount),
        mFutureVec(kSeastarCoreCount)
    {
    }
    virtual ~PsWorkAction()
    {}
    void Run()
    {
        for (int32_t i = 0; i < kSeastarCoreCount; ++i)
        {
            mFutureVec[i] = mSendQueue->Enqueue(item, i);
        }
        for (int32_t i = 0; i < kSeastarCoreCount; ++i)
        {
            mFutureVec[i].Get();
        }
    };
private:
    std::pair<ps::network::QueueHub<T>*, ps::network::QueueHub<T>*> mQueuePair;
    ps::network::QueueHub<T>* mSendQueue;
    ps::network::QueueHub<T>* mRecvQueue;
    const int32_t kSeastarCoreCount;
    vector<ps::network::Future<T>> mFutureVec;
};

} // namespace
} // namespace

#endif // ML_PS5_NETWORK_PS_QUEUE_WORK_ACTION_H

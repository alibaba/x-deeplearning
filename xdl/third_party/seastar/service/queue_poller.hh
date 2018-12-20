#ifndef PS_NETWORK_QUEUE_POLLER_HH_
#define PS_NETWORK_QUEUE_POLLER_HH_

#include <errno.h>
#include <iostream>
#include <unordered_set>
#include <vector>
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/queue_hub.hh"
#include "core/ps_common.hh"

extern uint64_t CoutCurrentTimeOnNanoSec(const char * prompt, bool output);

namespace ps
{
namespace network
{

class QueuePoller
{
public:
    QueuePoller(int32_t myCpuId, string queueName = "SEASTAR") :
        mQueueHubPair(ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>(queueName)),
        mInputHub(mQueueHubPair.first), mOutputHub(mQueueHubPair.second),
        kUserTheadCount(mInputHub->GetMCount()),
        kMyCpuId(myCpuId), mMetronome(0), mIsFree(true)
    {}

    ~QueuePoller() {}

    bool Poll()
    {
        bool hasWork = false;
        seastar::future<> f = this->DoPoll(hasWork);
        if (hasWork)
        {
	    mIsFree = false;
            f.then([this]() { this->Poll(); });
        }
        else
        {
            mIsFree = true;
        }

        return hasWork;
    }

    bool Available()
    {
        if (mIsFree)
        {
            // poll again
            Poll();
        }

        //return true;
        return false;
    }

    ps::network::QueueHub<ps::network::Item>* GetInputHub() const
    {
        return mInputHub;
    }

    ps::network::QueueHub<ps::network::Item>* GetOutputHub() const
    {
        return mOutputHub;
    }
private:
    seastar::future<> DoPoll(bool& hasWork)
    {
        hasWork = false;
        for (int32_t i = 0; i < kUserTheadCount; ++i)
        {
            ps::network::Item* item = NULL;
            if ((item = mInputHub->Dequeue(i, kMyCpuId)) != NULL)
            {
                hasWork = true;
                if (item->GetItemType() == ps::network::SeaStar)
                {
                    ps::network::SeastarWorkItem* seaStarItem
                        = static_cast<ps::network::SeastarWorkItem*>(item);
                    // every item can attach a callback.
                    if (seaStarItem->GetClosure() != NULL)
                    {
                        ps::service::seastar::ClosureManager::GetClosureMap()
                            .insert({seaStarItem->GetSequence(), seaStarItem->GetClosure()});
                    }
                    return seaStarItem->Run().then([this, item, i, seaStarItem]()
                    {
                        if (seaStarItem->GetNeedEnqueueBack())
                        {
                            mOutputHub->EnqueueBack(item, this->kMyCpuId, i);
                        }
                    });
                }
                else if (item->GetItemType() == ps::network::PsEngine)
                {
                    ps::network::PsWorkItem* psWorkItem = static_cast<ps::network::PsWorkItem*>(item);
                    return psWorkItem->Complete().then([psWorkItem]()
                            {
                            delete psWorkItem;
                            });
                }
            }
        }
        return seastar::make_ready_future<>();
    }

private:
    std::pair<ps::network::QueueHub<ps::network::Item>*,
        ps::network::QueueHub<ps::network::Item>*> mQueueHubPair;
    ps::network::QueueHub<ps::network::Item>* mInputHub;
    ps::network::QueueHub<ps::network::Item>* mOutputHub;
    const int32_t kUserTheadCount;
    const int32_t kMyCpuId;
    int32_t mMetronome;
    // indicate if Poll() is called
    bool mIsFree;
};

} // namespace network
} // namespace ps

#endif // PS_NETWORK_QUEUE_POLLER_HH_

#ifndef PS_ACTION_PUSH_ACTION_H
#define PS_ACTION_PUSH_ACTION_H

#include <iostream>
//#include <string>
//#include <memory>
//#include <exception>
//#include <vector>
//#include <mutex>
#include "core/ps_queue_hub/queue_hub_future.hh"
#include "core/ps_queue_hub/queue_hub.hh"
#include "service/client_network_context.hh"
#include "service/network_context_helper.hh"
#include "action/push_request_item.hh"
#include "action/pull_request_item.hh"

namespace ps
{
namespace action
{
using namespace std;

template<typename T>
class PushAction
{
public:
    PushAction(const vector<T>& v, int64_t begin, int64_t end) :
        mNetworkContext(ps::network::NetworkContextHelper::GetNetworkContext()),
        kSourceVec(v), kBegin(begin), kEnd(end),
        mQueuePair(ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("SEASTAR")),
        mSendQueue(mQueuePair.first),
        mRecvQueue(mQueuePair.second),
        kUserCount(mNetworkContext.GetUserThreadCount()),
        kServerCount(mNetworkContext.GetServerCount()),
        kSeastarCoreCount(mNetworkContext.GetCoresCount()),
        mFutureVec()
    {}
    virtual ~PushAction()
    {}
    void Run()
    {
        int32_t user = ps::network::QueueHub<ps::network::Item>::GetPortNumberOnThread();
        // TODO: use partitioner
        for (int32_t i = 0; i < kServerCount; ++i)
        {
            ps::action::PushRequestItem<T>* item
                = new PushRequestItem<T>(mNetworkContext,
                        user, kSourceVec, kBegin, kEnd, i);
            int32_t core = ps::network::ClientNetworkContext::GetCoreByServerId(i, kSeastarCoreCount);
            mFutureVec.push_back(mSendQueue->Enqueue(item, user, core));
        }
        for (int32_t i = 0; i < kServerCount; ++i)
        {
            mFutureVec[i].Get();
        }
    };
private:
    ps::network::ClientNetworkContext& mNetworkContext;
    const vector<T>& kSourceVec;
    const int64_t kBegin;
    const int64_t kEnd;
    std::pair<
        ps::network::QueueHub<ps::network::Item>*,
        ps::network::QueueHub<ps::network::Item>*> mQueuePair;
    ps::network::QueueHub<ps::network::Item>* mSendQueue;
    ps::network::QueueHub<ps::network::Item>* mRecvQueue;
    const int32_t kUserCount;
    const int32_t kServerCount;
    const int32_t kSeastarCoreCount;
    vector<ps::network::Future<ps::network::Item>> mFutureVec;
};

} // namespace
} // namespace

#endif // PS_ACTION_PUSH_ACTION_H

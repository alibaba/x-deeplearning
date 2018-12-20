#ifndef ML_PS5_NETWORK_SEASTAR_WORK_ITEM_H
#define ML_PS5_NETWORK_SEASTAR_WORK_ITEM_H

#include <iostream>
#include <string>
#include <memory>
#include <exception>
#include <vector>
#include <mutex>
#include "core/future.hh"
#include "core/ps_queue_hub/queue_work_item.hh"
#include "service/session_context.hh"
#include "service/seastar_status.hh"

namespace ps
{
namespace network
{
using namespace std;
class NetworkContext;

class SeastarWorkItem : public ps::network::Item
{
public:
    SeastarWorkItem(NetworkContext* networkContext, int32_t remoteId, bool needEnqueueBack = true)
        : ps::network::Item(SeaStar), mNetworkContext(networkContext), mRemoteId(remoteId), mNeedEnqueueBack(needEnqueueBack)
    {}
    virtual ~SeastarWorkItem()
    {}
    virtual seastar::future<> Run()
    {
        //std::cout << "SeastarWorkItem Run" << std::endl;
        return seastar::make_ready_future<>();
    }
    virtual void Complete()
    {
        //std::cout << "SeastarWorkItem Complete" << std::endl;
    };

    NetworkContext* GetNetworkContext() const
    {
        return mNetworkContext;
    }

    int32_t GetRemoteId() const
    {
        return mRemoteId;
    }

    bool GetNeedEnqueueBack() const {
        return mNeedEnqueueBack;
    }

    SessionContext* GetSessionContext() const;

private:
    NetworkContext* mNetworkContext;
    int32_t mRemoteId;  // Represent the remote server id
    bool mNeedEnqueueBack;
};

} // namespace
} // namespace

#endif // ML_PS5_NETWORK_SEASTAR_WORK_ITEM_H

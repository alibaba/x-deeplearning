#ifndef PS5_NETWORK_CONTEXT_HH_
#define PS5_NETWORK_CONTEXT_HH_

#include "core/ps_queue_hub/queue_hub.hh"
#include "core/ps_queue_hub/queue_hub_future.hh"
#include "service/session_context.hh"
#include "core/app-template.hh"
#include <vector>
#include <assert.h>

namespace ps
{
namespace network
{
using namespace std;

class NetworkContext
{
public:
    NetworkContext(int32_t serverCount, int32_t clientCount, int32_t coreCount, int32_t userThreadCount,
            string queueName = "SEASTAR") :
        mServerCount(serverCount), mClientCount(clientCount),
        mCoreCount(coreCount), mUserThreadCount(userThreadCount),
        mSessionsContext(),
        mQueueHubPair(ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>(queueName,
                    userThreadCount, coreCount)),
        mRunning(false)
    {}

    virtual ~NetworkContext() {}

    virtual void Reconnect()
    {
        assert("NetworkContext not impl Reconnect() !");
    }

    void SetRunning(bool r)
    {
        mRunning = r;
    }

    bool GetRunning() const
    {
        return mRunning;
    }

    int32_t GetServerCount() const
    {
        return mServerCount;
    }

    int32_t GetCoresCount() const
    {
        return mCoreCount;
    }

    int32_t GetClientCount() const
    {
        return mClientCount;
    }

    int32_t GetUserThreadCount() const
    {
        return mUserThreadCount;
    }

    virtual void SetSessionContextOfId(std::unique_ptr<SessionContext>&& sc, int64_t id)
    {
        assert("NetworkContext not impl SetSessionContextOfId(...) !");
    }

    virtual SessionContext* GetSessionOfId(int64_t id) const
    {
        assert("NetworkContext not impl GetSessionOfId(...) !");
        return NULL;
    }

    virtual std::unique_ptr<SessionContext>* GetSessionAddrOfId(int64_t id)
    {
        assert("NetworkContext not impl GetSessionAddrOfId(...) !");
        return NULL;
    }

    std::pair<ps::network::QueueHub<ps::network::Item>*, ps::network::QueueHub<ps::network::Item>*>
        GetQueueHubPair() const
    {
        return mQueueHubPair;
    }

protected:
    seastar::app_template mApp;
    int32_t mServerCount;
    int32_t mClientCount;
    int32_t mCoreCount;
    int32_t mUserThreadCount;
    std::vector<std::unique_ptr<SessionContext>> mSessionsContext;
    std::pair<ps::network::QueueHub<ps::network::Item>*,
        ps::network::QueueHub<ps::network::Item>*> mQueueHubPair;
    bool mRunning;
};

} // namespace network
} // namespace ps

#endif // PS5_NETWORK_CONTEXT_HH_

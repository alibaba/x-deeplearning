#include "core/reactor.hh"
#include "queue_hub.hh"

namespace ps
{
namespace network
{

// help function for client and server
// trick here: just for default Item type
std::pair<QueueHub<Item>*, QueueHub<Item>*> QueueHubFactory::GetDefaultQueueForClient()
{
    if (mHubMap.find("SEASTAR") == mHubMap.end())
    {
        return std::pair<QueueHub<Item>*, QueueHub<Item>*>(
                reinterpret_cast<QueueHub<Item>*>(NULL),
                reinterpret_cast<QueueHub<Item>*>(NULL));
    }
    return std::pair<QueueHub<Item>*, QueueHub<Item>*>(
        reinterpret_cast<QueueHub<Item>*>(mHubMap["SEASTAR"].first),
        reinterpret_cast<QueueHub<Item>*>(mHubMap["SEASTAR"].second));
}

std::pair<QueueHub<Item>*, QueueHub<Item>*> QueueHubFactory::GetDefaultQueueForServer()
{
    string queueName = "SEASTAR";
    if (seastar::smp::is_server_client_mode)
    {
        queueName = "SEASTAR2";
    }

    if (mHubMap.find(queueName) == mHubMap.end())
    {
        return std::pair<QueueHub<Item>*, QueueHub<Item>*>(
                reinterpret_cast<QueueHub<Item>*>(NULL),
                reinterpret_cast<QueueHub<Item>*>(NULL));
    }
    return std::pair<QueueHub<Item>*, QueueHub<Item>*>(
        reinterpret_cast<QueueHub<Item>*>(mHubMap[queueName].first),
        reinterpret_cast<QueueHub<Item>*>(mHubMap[queueName].second));
}

} // namespace ps
} // namespace network


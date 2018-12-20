#include "core/ps_queue_hub/queue_work_item.hh"
#include "service/network_context.hh"

namespace ps
{
namespace network
{
SessionContext* SeastarWorkItem::GetSessionContext() const
{
    return mNetworkContext->GetSessionOfId(mRemoteId);
}

} // namespace
} // namespace

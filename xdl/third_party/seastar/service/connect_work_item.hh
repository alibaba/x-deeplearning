#ifndef ML_PS5_NETWORK_CONNECT_WORK_ITEM_H
#define ML_PS5_NETWORK_CONNECT_WORK_ITEM_H

#include "core/ps_queue_hub/ps_work_item.hh"
#include "service/seastar_status.hh"
#include "service/seastar_exception.hh"

namespace ps
{
namespace network
{

class ConnectWorkItem : public PsWorkItem
{
public:
    ConnectWorkItem(bool reconnect = false) : mReconnect(reconnect) { }
    virtual ~ConnectWorkItem() { }

    virtual void Run()
    {
        // connect failed
        if (GetStatus() != Normal)
        {
            if (mReconnect)
            {
                throw ReconnectException();
            }
            else
            {
                throw ConnectException();
            }
        }
    }

    virtual seastar::future<> Complete()
    {
        return seastar::make_ready_future<>();
    };

private:
    bool mReconnect;
};

} // network
} // ps

#endif // ML_PS5_NETWORK_CONNECT_WORK_ITEM_H


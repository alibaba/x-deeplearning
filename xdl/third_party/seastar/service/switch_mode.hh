#ifndef ML_PS5_SWITCH_MODE_H
#define ML_PS5_SWITCH_MODE_H

#include "core/ps_queue_hub/seastar_work_item.hh"
#include "core/reactor.hh"

namespace ps
{
namespace network
{

class SwitchModeItem : public ps::network::SeastarWorkItem
{
public:
    SwitchModeItem(bool usePollMode)
    : ps::network::SeastarWorkItem(NULL, -1), mUsePollMode(usePollMode)
    {}

    seastar::future<> Run() override
    {
        if (mUsePollMode)
        {
	        seastar::engine().switch_to_poll_mode();
        }
        else
        {
            seastar::engine().switch_to_normal_mode();
        }

        return seastar::make_ready_future<>();
    }

private:
    bool mUsePollMode;
};

} // namespace net
} // ps

#endif // ML_PS5_SWITCH_MODE_H


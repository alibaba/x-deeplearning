#ifndef PS5_PACKET_TIMER_HH_
#define PS5_PACKET_TIMER_HH_

#include "core/distributed.hh"
#include "core/timer-set.hh"

using namespace seastar;
using namespace net;
using clock_type = lowres_clock;

namespace ps
{
namespace network
{

class ClientConnection;

class PacketTimer
{
public:
    // interval: ms
    PacketTimer(ClientConnection* cc, uint64_t interval = 1000)
        : mConn(cc), mInterval(interval) { }

    // start service
    void Start();

    // ms
    void SetTimeoutInterval(uint64_t interval)
    {
        mInterval = interval;
    }
private:
    ClientConnection* mConn;
    timer<> mTimer;
    uint64_t mInterval;
};

} // namespace network
} // namespace ps

#endif


#ifndef PS5_STATS_PRINTER_HH_
#define PS5_STATS_PRINTER_HH_

#include <limits>
#include "core/timer-set.hh"
#include "core/distributed.hh"

using namespace seastar;
using namespace net;
using clock_type = lowres_clock;

namespace ps
{
namespace network
{

uint64_t GetCurrentTimeInUs();

class SeastarClient;
class SeastarServer;

struct SystemStats
{
    uint32_t mCurrConnections {0};
    uint32_t mTotalConnections {0};
    uint64_t mEcho {0};
    uint32_t mMinRtt {std::numeric_limits<int>::max()};
    uint32_t mMaxRtt {0};
    uint64_t mReqCount {0};
    uint64_t mTotalRtt {0};

public:
    SystemStats();
    ~SystemStats() { }

    SystemStats Self();
    void operator+=(const SystemStats& other);
};

class ServerStatsPrinter {
private:
    timer<> mTimer;
    distributed<SeastarServer>& mShardServer;
    size_t mLast {0};
    uint64_t mMinQps {std::numeric_limits<int>::max()};
    uint64_t mMaxQps {0};

    seastar::future<SystemStats> Stats();

public:
    ServerStatsPrinter(seastar::distributed<SeastarServer>& s);
    void Start();
};

class ClientStatsPrinter {
private:
    timer<> mTimer;
    distributed<SeastarClient>& mSeastarClient;
    uint32_t mMinRtt {std::numeric_limits<int>::max()};
    uint32_t mMaxRtt {0};
    uint64_t mReqCount {0};
    uint64_t mTotalRtt {0};

    seastar::future<SystemStats> Stats();

public:
    ClientStatsPrinter(seastar::distributed<SeastarClient>& c);
    void Start();
};

} // namespace network
} // namespace ps

#endif // PS5_STATS_PRINTER_HH_

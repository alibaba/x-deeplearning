#include <sys/time.h>
#include "service/server.hh"
#include "service/client.hh"
#include "stats_printer.hh"

namespace ps
{
namespace network
{

uint64_t GetCurrentTimeInUs()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

SystemStats::SystemStats()
{
}

SystemStats SystemStats::Self()
{
    return *this;
}

void SystemStats::operator+=(const SystemStats& other)
{
    mCurrConnections += other.mCurrConnections;
    mTotalConnections += other.mTotalConnections;
    mEcho += other.mEcho;
    
    mMinRtt = std::min(mMinRtt, other.mMinRtt);
    mMaxRtt = std::max(mMaxRtt, other.mMaxRtt);
    mReqCount += other.mReqCount;
    mTotalRtt += other.mTotalRtt;
}

// -------------------------------------

ServerStatsPrinter::ServerStatsPrinter(seastar::distributed<SeastarServer>& s) : mShardServer(s) { }

seastar::future<SystemStats> ServerStatsPrinter::Stats()
{
    return mShardServer.map_reduce(adder<SystemStats>(), &SeastarServer::Stats);
}

void ServerStatsPrinter::Start()
{
    mTimer.set_callback([this]
    {
        Stats().then([this] (auto stats)
        {
            uint64_t qps = stats.mEcho - mLast;
            mMinQps = mMinQps < qps ? mMinQps : qps;
            mMaxQps = mMaxQps > qps ? mMaxQps : qps;
            mLast = stats.mEcho;

            std::cout << " current " << stats.mCurrConnections 
                      << " total " << stats.mTotalConnections 
                      << " cur qps " << qps 
                      << " max qps " << mMaxQps
                      << " min qps " << mMinQps
                      << "\n";
        });
    });
    // per second
    mTimer.arm_periodic(std::chrono::seconds(1));
}

// -------------------------------------

ClientStatsPrinter::ClientStatsPrinter(seastar::distributed<SeastarClient>& c) : mSeastarClient(c) { }

seastar::future<SystemStats> ClientStatsPrinter::Stats()
{
    return mSeastarClient.map_reduce(adder<SystemStats>(), &SeastarClient::Stats);
}

void ClientStatsPrinter::Start()
{
    mTimer.set_callback([this]
    {
        Stats().then([this] (auto stats)
        {
            mMinRtt = stats.mMinRtt;
            mMaxRtt = stats.mMaxRtt;
            mReqCount = stats.mReqCount;
            mTotalRtt = stats.mTotalRtt;

            std::cout << " total count " << mReqCount
		      << " max rtt " << mMaxRtt
                      << " min rtt " << mMinRtt
                      << " avg rtt " << mTotalRtt / (mReqCount == 0 ? 1 : mReqCount)
                      << "\n";
        });
    });
    // per second
    mTimer.arm_periodic(std::chrono::seconds(1));
}

} // namespace network
} // namespace ps

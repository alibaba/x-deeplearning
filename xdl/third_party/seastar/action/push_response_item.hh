#ifndef PS_ACTION_ACTION_PUSH_RESPONSE_ITEM_H
#define PS_ACTION_ACTION_PUSH_RESPONSE_ITEM_H

#include <stdint.h>
#include <iostream>
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "service/session_context.hh"

namespace ps
{
namespace action
{
using namespace std;

class PushResponseItem : public ps::network::PsWorkItem
{
public:
    PushResponseItem(const ps::coding::MessageHeader::TimeStamp& ts)
        : ps::network::PsWorkItem(), mHeaderTimeStamp(ts)
    {}

    void Run() override
    {
        mHeaderTimeStamp.mWorkerEnqueueTime2 = this->mEnqueueTime;
        mHeaderTimeStamp.mWorkerDequeueTime2 = this->mDequeueTime;
        if (sCurrentPreTimes < kPreTimes)
        {
            ++sCurrentPreTimes;
            //cout << "Skip Prepare Message." << endl;
            //cout << "ps::coding::unittest::WorkerResponseItem<double>::kPreTimes="
                //<< ps::coding::unittest::WorkerResponseItem<double>::kPreTimes << endl;
            //cout << "sCurrentPreTimes=" << sCurrentPreTimes << endl;
            //cout << "2:" << mHeaderTimeStamp.mServerProcessTime
                //<< "___" << mHeaderTimeStamp.mWorkerSessionWriteTime << endl;
            //cout << "8:" <<  mHeaderTimeStamp.mWorkerProcessTime
                //<< "___" << mHeaderTimeStamp.mServerSessionWriteTime << endl;
            return;
        }
        //mHeaderTimeStamp.Output();
        sStatVec[0] += mHeaderTimeStamp.mWorkerDequeueTime - mHeaderTimeStamp.mWorkerEnqueueTime;
        sMaxVec[0] = max<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime - mHeaderTimeStamp.mWorkerEnqueueTime, sMaxVec[0]);
        sMinVec[0] = min<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime - mHeaderTimeStamp.mWorkerEnqueueTime, sMinVec[0]);
        sStatVec[1] += mHeaderTimeStamp.mWorkerSessionWriteTime - mHeaderTimeStamp.mWorkerDequeueTime;
        sMaxVec[1] = max<int64_t>(
                mHeaderTimeStamp.mWorkerSessionWriteTime - mHeaderTimeStamp.mWorkerDequeueTime, sMaxVec[1]);
        sMinVec[1] = min<int64_t>(
                mHeaderTimeStamp.mWorkerSessionWriteTime - mHeaderTimeStamp.mWorkerDequeueTime, sMinVec[1]);
        sStatVec[2] += mHeaderTimeStamp.mServerProcessTime - mHeaderTimeStamp.mWorkerSessionWriteTime;
        sMaxVec[2] = max<int64_t>(
                mHeaderTimeStamp.mServerProcessTime - mHeaderTimeStamp.mWorkerSessionWriteTime, sMaxVec[2]);
        sMinVec[2] = min<int64_t>(
                mHeaderTimeStamp.mServerProcessTime - mHeaderTimeStamp.mWorkerSessionWriteTime, sMinVec[2]);
        sStatVec[3] += mHeaderTimeStamp.mServerEnqueueTime - mHeaderTimeStamp.mServerProcessTime;
        sMaxVec[3] = max<int64_t>(
                mHeaderTimeStamp.mServerEnqueueTime - mHeaderTimeStamp.mServerProcessTime, sMaxVec[3]);
        sMinVec[3] = min<int64_t>(
                mHeaderTimeStamp.mServerEnqueueTime - mHeaderTimeStamp.mServerProcessTime, sMinVec[3]);
        sStatVec[4] += mHeaderTimeStamp.mServerDequeueTime - mHeaderTimeStamp.mServerEnqueueTime;
        sMaxVec[4] = max<int64_t>(
                mHeaderTimeStamp.mServerDequeueTime - mHeaderTimeStamp.mServerEnqueueTime, sMaxVec[4]);
        sMinVec[4] = min<int64_t>(
                mHeaderTimeStamp.mServerDequeueTime - mHeaderTimeStamp.mServerEnqueueTime, sMinVec[4]);
        sStatVec[5] += mHeaderTimeStamp.mServerEnqueueTime2 - mHeaderTimeStamp.mServerDequeueTime;
        sMaxVec[5] = max<int64_t>(
                mHeaderTimeStamp.mServerEnqueueTime2 - mHeaderTimeStamp.mServerDequeueTime, sMaxVec[5]);
        sMinVec[5] = min<int64_t>(
                mHeaderTimeStamp.mServerEnqueueTime2 - mHeaderTimeStamp.mServerDequeueTime, sMinVec[5]);
        sStatVec[6] += mHeaderTimeStamp.mServerDequeueTime2 - mHeaderTimeStamp.mServerEnqueueTime2;
        sMaxVec[6] = max<int64_t>(
                mHeaderTimeStamp.mServerDequeueTime2 - mHeaderTimeStamp.mServerEnqueueTime2, sMaxVec[6]);
        sMinVec[6] = min<int64_t>(
                mHeaderTimeStamp.mServerDequeueTime2 - mHeaderTimeStamp.mServerEnqueueTime2, sMinVec[6]);
        sStatVec[7] += mHeaderTimeStamp.mServerSessionWriteTime - mHeaderTimeStamp.mServerDequeueTime2;
        sMaxVec[7] = max<int64_t>(
                mHeaderTimeStamp.mServerSessionWriteTime - mHeaderTimeStamp.mServerDequeueTime2, sMaxVec[7]);
        sMinVec[7] = min<int64_t>(
                mHeaderTimeStamp.mServerSessionWriteTime - mHeaderTimeStamp.mServerDequeueTime2, sMinVec[7]);
        sStatVec[8] += mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSessionWriteTime;
        sMaxVec[8] = max<int64_t>(
                mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSessionWriteTime, sMaxVec[8]);
        sMinVec[8] = min<int64_t>(
                mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSessionWriteTime, sMinVec[8]);
        sStatVec[9] += mHeaderTimeStamp.mWorkerEnqueueTime2 - mHeaderTimeStamp.mWorkerProcessTime;
        sMaxVec[9] = max<int64_t>(
                mHeaderTimeStamp.mWorkerEnqueueTime2 - mHeaderTimeStamp.mWorkerProcessTime, sMaxVec[9]);
        sMinVec[9] = min<int64_t>(
                mHeaderTimeStamp.mWorkerEnqueueTime2 - mHeaderTimeStamp.mWorkerProcessTime, sMinVec[9]);
        sStatVec[10] += mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime2;
        sMaxVec[10] = max<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime2, sMaxVec[10]);
        sMinVec[10] = min<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime2, sMinVec[10]);
        sStatVec[11] += mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime;
        sMaxVec[11] = max<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime, sMaxVec[11]);
        sMinVec[11] = min<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime, sMinVec[11]);

        //cout << "mHeaderTimeStamp.mServerSessionWriteTime:" << mHeaderTimeStamp.mServerSessionWriteTime << endl;
        //cout << "mHeaderTimeStamp.mServerSendMsgTime:" << mHeaderTimeStamp.mServerSendMsgTime << endl;
        //cout << "mHeaderTimeStamp.mWorkerProcessTime:" << mHeaderTimeStamp.mWorkerProcessTime << endl;
        sStatVec[12] += mHeaderTimeStamp.mServerSendMsgTime - mHeaderTimeStamp.mServerSessionWriteTime;
        sMaxVec[12] = max<int64_t>(
                mHeaderTimeStamp.mServerSendMsgTime - mHeaderTimeStamp.mServerSessionWriteTime, sMaxVec[12]);
        sMinVec[12] = min<int64_t>(
                mHeaderTimeStamp.mServerSendMsgTime - mHeaderTimeStamp.mServerSessionWriteTime, sMinVec[12]);
        sStatVec[13] += mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSendMsgTime;
        sMaxVec[13] = max<int64_t>(
                mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSendMsgTime, sMaxVec[13]);
        sMinVec[13] = min<int64_t>(
                mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSendMsgTime, sMinVec[13]);


        return;
    }
    seastar::future<> Complete()
    {
        return seastar::make_ready_future<>();
    }
public:
    static const int32_t kPreTimes = 64;
    static int32_t sCurrentPreTimes;
    static std::vector<int64_t> sStatVec;
    static std::vector<int64_t> sMaxVec;
    static std::vector<int64_t> sMinVec;
private:
    ps::coding::MessageHeader::TimeStamp mHeaderTimeStamp;
};

} // namespace
} // namespace

#endif // PS_ACTION_ACTION_PUSH_RESPONSE_ITEM_H

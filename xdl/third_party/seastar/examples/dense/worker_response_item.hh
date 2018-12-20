#ifndef ML_PS5_EXAMPLES_DENSE_WORKER_RESPONSE_ITEM_H
#define ML_PS5_EXAMPLES_DENSE_WORKER_RESPONSE_ITEM_H

#include <stdint.h>
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "core/ps_coding/message_serializer.hh"
#include "examples/dense/echo_request_serializer.hh"
#include "service/session_context.hh"

extern std::vector<int64_t> gStatVec;
extern std::vector<int64_t> gMaxVec;
extern std::vector<int64_t> gMinVec;

namespace ps
{
namespace coding
{
namespace unittest
{
using namespace std;

template<typename T>
class WorkerResponseItem : public ps::network::PsWorkItem
{
public:
    WorkerResponseItem(ps::coding::TemporaryBuffer&& t, const ps::coding::MessageHeader::TimeStamp& ts)
        : ps::network::PsWorkItem(), mTempBuffer(move(t)), mHeaderTimeStamp(ts)
    {}

    void Run() override
    {
        //cout << "WorkerResponseItem Run, work on worker" << endl;
        //const int32_t count = mTempBuffer.size() / sizeof(T);
        //cout << "WorkerResponseItem,count=" << count << endl;
        //const T* ptr = reinterpret_cast<const T*>(mTempBuffer.get());
        //for (int32_t i = 0; i < count; ++i)
        //{
            //cout << *(ptr + i) << " ";
        //}
        //cout << endl;
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
        gStatVec[0] += mHeaderTimeStamp.mWorkerDequeueTime - mHeaderTimeStamp.mWorkerEnqueueTime;
        gMaxVec[0] = max<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime - mHeaderTimeStamp.mWorkerEnqueueTime, gMaxVec[0]);
        gMinVec[0] = min<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime - mHeaderTimeStamp.mWorkerEnqueueTime, gMinVec[0]);
        gStatVec[1] += mHeaderTimeStamp.mWorkerSessionWriteTime - mHeaderTimeStamp.mWorkerDequeueTime;
        gMaxVec[1] = max<int64_t>(
                mHeaderTimeStamp.mWorkerSessionWriteTime - mHeaderTimeStamp.mWorkerDequeueTime, gMaxVec[1]);
        gMinVec[1] = min<int64_t>(
                mHeaderTimeStamp.mWorkerSessionWriteTime - mHeaderTimeStamp.mWorkerDequeueTime, gMinVec[1]);
        gStatVec[2] += mHeaderTimeStamp.mServerProcessTime - mHeaderTimeStamp.mWorkerSessionWriteTime;
        gMaxVec[2] = max<int64_t>(
                mHeaderTimeStamp.mServerProcessTime - mHeaderTimeStamp.mWorkerSessionWriteTime, gMaxVec[2]);
        gMinVec[2] = min<int64_t>(
                mHeaderTimeStamp.mServerProcessTime - mHeaderTimeStamp.mWorkerSessionWriteTime, gMinVec[2]);
        gStatVec[3] += mHeaderTimeStamp.mServerEnqueueTime - mHeaderTimeStamp.mServerProcessTime;
        gMaxVec[3] = max<int64_t>(
                mHeaderTimeStamp.mServerEnqueueTime - mHeaderTimeStamp.mServerProcessTime, gMaxVec[3]);
        gMinVec[3] = min<int64_t>(
                mHeaderTimeStamp.mServerEnqueueTime - mHeaderTimeStamp.mServerProcessTime, gMinVec[3]);
        gStatVec[4] += mHeaderTimeStamp.mServerDequeueTime - mHeaderTimeStamp.mServerEnqueueTime;
        gMaxVec[4] = max<int64_t>(
                mHeaderTimeStamp.mServerDequeueTime - mHeaderTimeStamp.mServerEnqueueTime, gMaxVec[4]);
        gMinVec[4] = min<int64_t>(
                mHeaderTimeStamp.mServerDequeueTime - mHeaderTimeStamp.mServerEnqueueTime, gMinVec[4]);
        gStatVec[5] += mHeaderTimeStamp.mServerEnqueueTime2 - mHeaderTimeStamp.mServerDequeueTime;
        gMaxVec[5] = max<int64_t>(
                mHeaderTimeStamp.mServerEnqueueTime2 - mHeaderTimeStamp.mServerDequeueTime, gMaxVec[5]);
        gMinVec[5] = min<int64_t>(
                mHeaderTimeStamp.mServerEnqueueTime2 - mHeaderTimeStamp.mServerDequeueTime, gMinVec[5]);
        gStatVec[6] += mHeaderTimeStamp.mServerDequeueTime2 - mHeaderTimeStamp.mServerEnqueueTime2;
        gMaxVec[6] = max<int64_t>(
                mHeaderTimeStamp.mServerDequeueTime2 - mHeaderTimeStamp.mServerEnqueueTime2, gMaxVec[6]);
        gMinVec[6] = min<int64_t>(
                mHeaderTimeStamp.mServerDequeueTime2 - mHeaderTimeStamp.mServerEnqueueTime2, gMinVec[6]);
        gStatVec[7] += mHeaderTimeStamp.mServerSessionWriteTime - mHeaderTimeStamp.mServerDequeueTime2;
        gMaxVec[7] = max<int64_t>(
                mHeaderTimeStamp.mServerSessionWriteTime - mHeaderTimeStamp.mServerDequeueTime2, gMaxVec[7]);
        gMinVec[7] = min<int64_t>(
                mHeaderTimeStamp.mServerSessionWriteTime - mHeaderTimeStamp.mServerDequeueTime2, gMinVec[7]);
        gStatVec[8] += mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSessionWriteTime;
        gMaxVec[8] = max<int64_t>(
                mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSessionWriteTime, gMaxVec[8]);
        gMinVec[8] = min<int64_t>(
                mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSessionWriteTime, gMinVec[8]);
        gStatVec[9] += mHeaderTimeStamp.mWorkerEnqueueTime2 - mHeaderTimeStamp.mWorkerProcessTime;
        gMaxVec[9] = max<int64_t>(
                mHeaderTimeStamp.mWorkerEnqueueTime2 - mHeaderTimeStamp.mWorkerProcessTime, gMaxVec[9]);
        gMinVec[9] = min<int64_t>(
                mHeaderTimeStamp.mWorkerEnqueueTime2 - mHeaderTimeStamp.mWorkerProcessTime, gMinVec[9]);
        gStatVec[10] += mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime2;
        gMaxVec[10] = max<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime2, gMaxVec[10]);
        gMinVec[10] = min<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime2, gMinVec[10]);
        gStatVec[11] += mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime;
        gMaxVec[11] = max<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime, gMaxVec[11]);
        gMinVec[11] = min<int64_t>(
                mHeaderTimeStamp.mWorkerDequeueTime2 - mHeaderTimeStamp.mWorkerEnqueueTime, gMinVec[11]);


        //cout << "mHeaderTimeStamp.mServerSessionWriteTime:" << mHeaderTimeStamp.mServerSessionWriteTime << endl;
        //cout << "mHeaderTimeStamp.mServerSendMsgTime:" << mHeaderTimeStamp.mServerSendMsgTime << endl;
        //cout << "mHeaderTimeStamp.mWorkerProcessTime:" << mHeaderTimeStamp.mWorkerProcessTime << endl;
        gStatVec[12] += mHeaderTimeStamp.mServerSendMsgTime - mHeaderTimeStamp.mServerSessionWriteTime;
        gMaxVec[12] = max<int64_t>(
                mHeaderTimeStamp.mServerSendMsgTime - mHeaderTimeStamp.mServerSessionWriteTime, gMaxVec[12]);
        gMinVec[12] = min<int64_t>(
                mHeaderTimeStamp.mServerSendMsgTime - mHeaderTimeStamp.mServerSessionWriteTime, gMinVec[12]);
        gStatVec[13] += mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSendMsgTime;
        gMaxVec[13] = max<int64_t>(
                mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSendMsgTime, gMaxVec[13]);
        gMinVec[13] = min<int64_t>(
                mHeaderTimeStamp.mWorkerProcessTime - mHeaderTimeStamp.mServerSendMsgTime, gMinVec[13]);


        return;
    }
    seastar::future<> Complete()
    {
        //cout << "WorkerResponseItem Complete, work on server, nothing to do." << endl;
        return seastar::make_ready_future<>();
    }
public:
    static const int32_t kPreTimes = 64;
    static int32_t sCurrentPreTimes;
private:
    ps::coding::TemporaryBuffer mTempBuffer;
    ps::coding::MessageHeader::TimeStamp mHeaderTimeStamp;
};

template<typename T> int32_t ps::coding::unittest::WorkerResponseItem<T>::sCurrentPreTimes = 0;

} // namespace unittest
} // namespace coding
} // ps

#endif // ML_PS5_EXAMPLES_DENSE_WORKER_RESPONSE_ITEM_H

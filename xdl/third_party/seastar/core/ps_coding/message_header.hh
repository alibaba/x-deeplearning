#ifndef ML_PS5_CODING_MESSAGE_HEADER_H
#define ML_PS5_CODING_MESSAGE_HEADER_H

#include <iostream>
#include <stddef.h>
#include <stdint.h>

namespace ps
{
namespace coding
{

struct MessageHeader
{
    size_t mMetaBufferSize;
    size_t mDataBufferSize;
    uint64_t mProcessorClassId;
    uint64_t mSequence;
    int32_t mUserThreadId;
    int32_t mReserved;
    uint64_t mCalculateCostTime; // for performence test

#ifdef USE_STATISTICS
    struct TimeStamp
    {
        TimeStamp() :
            mWorkerEnqueueTime(0), mWorkerDequeueTime(0), mWorkerSessionWriteTime(0),
            mWorkerStreamWriteTime(0), mServerProcessTime(0), mServerEnqueueTime(0),
            mServerDequeueTime(0), mServerEnqueueTime2(0), mServerDequeueTime2(0),
            mServerSessionWriteTime(0), mServerStreamWriteTime(0), mWorkerProcessTime(0),
            mWorkerEnqueueTime2(0), mWorkerDequeueTime2(0), mWorkerSendMsgTime(0), mServerSendMsgTime(0)
        {}
        uint64_t mWorkerEnqueueTime;
        uint64_t mWorkerDequeueTime;
        uint64_t mWorkerSessionWriteTime;
        uint64_t mWorkerStreamWriteTime;
        uint64_t mServerProcessTime;
        uint64_t mServerEnqueueTime;
        uint64_t mServerDequeueTime;
        uint64_t mServerEnqueueTime2;
        uint64_t mServerDequeueTime2;
        uint64_t mServerSessionWriteTime;
        uint64_t mServerStreamWriteTime;
        uint64_t mWorkerProcessTime;
        uint64_t mWorkerEnqueueTime2;
        uint64_t mWorkerDequeueTime2;
        uint64_t mWorkerSendMsgTime;
        uint64_t mServerSendMsgTime;

        void Output()
        {
            std::cout << "mWorkerDequeueTime-mWorkerEnqueueTime="
                << mWorkerDequeueTime - mWorkerEnqueueTime << std::endl;
            std::cout << "mWorkerSessionWriteTime-mWorkerDequeueTime="
                << mWorkerSessionWriteTime - mWorkerDequeueTime << std::endl;
            //std::cout << "mWorkerStreamWriteTime-mWorkerSessionWriteTime="
                //<< mWorkerStreamWriteTime - mWorkerSessionWriteTime << std::endl;
            //std::cout << "mServerProcessTime-mWorkerStreamWriteTime="
                //<< mServerProcessTime - mWorkerStreamWriteTime << std::endl;
            std::cout << "mServerProcessTime-mWorkerSessionWriteTime="
                << mServerProcessTime - mWorkerSessionWriteTime << std::endl;
            std::cout << "mServerEnqueueTime-mServerProcessTime="
                << mServerEnqueueTime-mServerProcessTime << std::endl;
            std::cout << "mServerDequeueTime-mServerEnqueueTime="
                << mServerDequeueTime - mServerEnqueueTime << std::endl;
            std::cout << "mServerEnqueueTime2-mServerDequeueTime="
                << mServerEnqueueTime2 - mServerDequeueTime << std::endl;
            std::cout << "mServerDequeueTime2-mServerEnqueueTime2="
                << mServerDequeueTime2 - mServerEnqueueTime2 << std::endl;
            std::cout << "mServerSessionWriteTime-mServerDequeueTime2="
                << mServerSessionWriteTime - mServerDequeueTime2 << std::endl;
            //std::cout << "mServerStreamWriteTime-mServerSessionWriteTime="
                //<< mServerStreamWriteTime - mServerSessionWriteTime << std::endl;
            //std::cout << "mWorkerProcessTime-mServerStreamWriteTime="
                //<< mWorkerProcessTime - mServerStreamWriteTime << std::endl;
            std::cout << "mWorkerProcessTime-mServerSessionWriteTime="
                << mWorkerProcessTime - mServerSessionWriteTime << std::endl;
            std::cout << "mWorkerEnqueueTime2-mWorkerProcessTime="
                << mWorkerEnqueueTime2 - mWorkerProcessTime << std::endl;
            std::cout << "mWorkerDequeueTime2-mWorkerEnqueueTime2="
                << mWorkerDequeueTime2 - mWorkerEnqueueTime2 << std::endl;
        }
    } mTimeStamp;

#endif

    MessageHeader()
    {
        mMetaBufferSize = 0;
        mDataBufferSize = 0;
        mProcessorClassId = 0;
        mSequence = 0;
        mUserThreadId = -1;
        mReserved = 0;
        mCalculateCostTime = 0;
    }
};

} // namespace coding
} // namespace ps

#endif // ML_PS5_CODING_MESSAGE_HEADER_H

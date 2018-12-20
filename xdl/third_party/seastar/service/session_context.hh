#ifndef SESSION_CONTEXT_HH_
#define SESSION_CONTEXT_HH_

#include "core/future.hh"
#include "core/ps_coding/message_serializer.hh"
#include <assert.h>

namespace ps
{
namespace network
{

class BaseConnection;

class SessionContext
{
public:
    SessionContext(BaseConnection* conn) : mConn(conn), mProcessorId(0)
        , mIsBadStatus(false), mHaveTimeOutPacket(false) {}
    virtual seastar::future<> Write(ps::coding::MessageSerializer* serializer) = 0;
    virtual ~SessionContext() { }

    virtual seastar::future<> Reconnect(std::string serverAddr, int64_t remoteId, 
        int64_t userThreadId, uint64_t sequenceId)
    {
        assert("Session Context not impl Reconnect(...)! ");
        return seastar::make_ready_future();
    }

    BaseConnection* GetConnection() const
    {
        return mConn;
    }

    void SetProcessorId(uint64_t id)
    {
        mProcessorId = id;
    }

    // A special function: compatible with both methods
    // 1. todo exception or timeout packet via MxN queue
    // 2. todo ... in processor which in seastar thread
    // if user set the mProcessorId, that indicate use method 2
    uint64_t GetProcessorId() const
    {
        return mProcessorId;
    }

    void SetBadStatus(bool b)
    {
        mIsBadStatus = b;
    }

    bool GetIsBadStatus() const
    {
        return mIsBadStatus;
    }

    void ClearSavedSequence()
    {
        mSavedSequenceWhenBroken.clear();
    }

    const std::vector<uint64_t>& GetSavedSequenceWhenBroken() const
    {
        return mSavedSequenceWhenBroken;
    }

    void PutOneToSavedSequence(uint64_t seq)
    {
        mSavedSequenceWhenBroken.push_back(seq);
    }

    void SetHaveTimeOutPacket(bool b)
    {
        mHaveTimeOutPacket = b;
    }

    bool GetHaveTimeOutPacket() const
    {
        return mHaveTimeOutPacket;
    }

protected:
    BaseConnection* mConn;

private:
    // will be used when get a exception.
    // if mProcesserId = 0, exception will be 
    // send to user by MxN queue
    uint64_t mProcessorId;

    // mIsBadStatus is set to true
    // when connection is broken
    bool mIsBadStatus;

    // save all sequence id when connection is broken
    // todo: will use these sequences in processor
    // when encounter exception or packet timeout
    std::vector<uint64_t> mSavedSequenceWhenBroken;

    // if there are some timeout packets when timeout
    // timer alert.
    bool mHaveTimeOutPacket;
};

} // namesapce network
} // namesapce ps

#endif /*SESSION_CONTEXT_HH_*/

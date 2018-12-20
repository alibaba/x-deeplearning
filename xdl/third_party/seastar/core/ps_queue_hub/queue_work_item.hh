#ifndef ML_PS5_NETWORK_QUEUE_WORK_ITEM_H
#define ML_PS5_NETWORK_QUEUE_WORK_ITEM_H

#include <iostream>
#include <string>
#include <memory>
#include <exception>
#include <vector>
#include <mutex>
#include "service/seastar_status.hh"
#include "core/ps_common.hh"

namespace ps
{
namespace network
{
using namespace std;

enum ItemType
{
    SeaStar = 0, // The item is run in SeaStar engine
    PsEngine = 1 // The item is run in PS engine
};

class Item
{
public:
    Item(ItemType type) : mEnqueueTime(0), mDequeueTime(0),
        kItemType(type), mDeprecated(false), mSequence(0), mSideSequ(0),
        mLockSequence(false), mStatus(Normal), mErrno(-1), mServerId(-1),
        mClosure(NULL)
    {
    }
    virtual ~Item() {}
    ItemType GetItemType() const
    {
        return kItemType;
    }

    void Deprecate()
    {
        mDeprecated = true;
    }
    void Finish()
    {
        mSequence = 0;
    }
    bool IsDeprecated() const
    {
        return mDeprecated;
    }
    bool IsDeprecatedAndFinished() const
    {
        return mDeprecated && mSequence == 0;
    }
    bool IsSequMatched(uint64_t s) const
    {
        return mSequence == s;
    }
    void SetSequence(uint64_t s)
    {
        if (mLockSequence) { return; }
        mSequence = s;
        mLockSequence = true;
    }
    uint64_t GetSequence() const
    {
        return mSequence;
    }
    void SetSideSequ(uint64_t s)
    {
        mSideSequ = s;
    }
    uint64_t GetSideSequ() const
    {
        return mSideSequ;
    }
    void SetStatus(ConnectionStatus status)
    {
        mStatus = status;
    }
    ConnectionStatus GetStatus() const
    {
        return mStatus;
    }
    void SetErrno(int32_t eno)
    {
        mErrno = eno;
    }
    int32_t GetErrno() const
    {
        return mErrno;
    }
    void SetServerId(int64_t id)
    {
        mServerId = id;
    }
    int64_t GetServerId() const
    {
        return mServerId;
    }
    void SetClosure(ps::service::seastar::Closure* cs)
    {
        mClosure = cs;
    }
    ps::service::seastar::Closure* GetClosure() const
    {
        return mClosure;
    }
public:
    uint64_t mEnqueueTime;
    uint64_t mDequeueTime;
protected:
    const ItemType kItemType;
private:
    bool mDeprecated;
    uint64_t mSequence;
    uint64_t mSideSequ;
    bool mLockSequence;
    ConnectionStatus mStatus;
    int32_t mErrno;
    int64_t mServerId;
    ps::service::seastar::Closure* mClosure;
};

} // namespace
} // namespace

#endif // ML_PS5_NETWORK_QUEUE_WORK_ITEM_H

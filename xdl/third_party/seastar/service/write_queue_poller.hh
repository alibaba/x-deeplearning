#ifndef PS_NETWORK_WRITE_QUEUE_POLLER_HH_
#define PS_NETWORK_WRITE_QUEUE_POLLER_HH_

#include "core/ps_queue_hub/readerwriterqueue.hh"
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/queue_hub.hh"
#include "service/common.hh"

namespace ps
{
namespace network
{

struct InterItem
{
    ps::network::SessionContext* sc;
    ps::coding::MessageSerializer* serializer;
};

class WriteQueuePoller
{
public:
    WriteQueuePoller() : mIsFree(true)
    {
        mQueue = new moodycamel::ReaderWriterQueue<InterItem>(65535);
    }

    ~WriteQueuePoller()
    {
        delete mQueue;
    }

    void Enqueue(ps::network::SessionContext* sc, ps::coding::MessageSerializer* serializer)
    {
        while(!mQueue->try_enqueue(InterItem{sc, serializer}));
    }

    void Poll()
    {
        bool hasItem = false;
        seastar::future<> f = this->DoPoll(hasItem);
        if (hasItem)
        {
            mIsFree = false;
            f.then([this]() { this->Poll(); });
        }
        else
        {
	        mIsFree = true;
        }
    }

    seastar::future<> DoPoll(bool& hasItem)
    {
        InterItem item;
        if (mQueue->try_dequeue(item))
        {
            hasItem = true;
            if (!ServerConnctionManager::IsAlive(item.sc))
            {
                return seastar::make_ready_future<>();
            }
            return item.sc->Write(item.serializer);
        }
        return seastar::make_ready_future<>();
    }

    bool Available()
    {
        if (mIsFree)
        {
            Poll();
        }
        return false;
    }
private:
    moodycamel::ReaderWriterQueue<InterItem>* mQueue;
    bool mIsFree;
};

} // namespace network
} // namespace ps

#endif

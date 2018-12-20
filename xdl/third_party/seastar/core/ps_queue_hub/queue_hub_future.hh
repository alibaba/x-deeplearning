#ifndef ML_PS5_NETWORK_QUEUE_HUB_FUTURE_H
#define ML_PS5_NETWORK_QUEUE_HUB_FUTURE_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include "core/ps_queue_hub/readerwriterqueue.hh"
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/seastar_work_item.hh"
#include "core/ps_queue_hub/ps_work_item.hh"

namespace ps
{
namespace network
{
using namespace std;

template<typename T>
class Future
{
public:
    Future(moodycamel::ReaderWriterQueue<T*>* q, T* item,
            moodycamel::ReaderWriterQueue<T*>* p = NULL,
            unordered_map<uint64_t, T*>* m = NULL)
        : mQueue(q), mMyItem(item), mSendQ(p), mMap(m)
    {}
    ~Future()
    {}
    void Get()
    {
        if (mMyItem == NULL)
        {
            T* ptr = NULL;
            if (mQueue->try_dequeue(ptr))
            {
                if (ptr->GetItemType() == ps::network::PsEngine)
                {
                    ps::network::PsWorkItem *item = static_cast<ps::network::PsWorkItem*>(ptr);
                    item->Run();
                    if (mSendQ != NULL)
                    {
                        while(!mSendQ->try_enqueue(item)) {}
                    }
                }
            }
            return;
        }
        if (mMyItem->IsDeprecatedAndFinished())
        {
            // Early dequeue has made this item deprecated and finished.
            mMap->erase(mMyItem->GetSequence());
            delete mMyItem;
            mMyItem = NULL;
            return;
        }
        while (true)
        {
            T* ptr = NULL;
            while(!mQueue->try_dequeue(ptr)){}

            uint64_t sideSequ = ptr->GetSideSequ();
            if (sideSequ > 0)
            {
                // This is a side request
                if (ptr->GetItemType() == ps::network::PsEngine)
                {
                    ps::network::PsWorkItem* psItem = static_cast<ps::network::PsWorkItem*>(ptr);
                    psItem->Run();
                    if (mSendQ != NULL) { while(!mSendQ->try_enqueue(ptr)) {} }
                }
                if (mMyItem->IsSequMatched(sideSequ))
                {
                    mMyItem->Finish(); // Don't need to monitor this sequence
                    if (mMyItem->IsDeprecatedAndFinished())
                    {
                        mMap->erase(sideSequ);
                        delete mMyItem;
                        mMyItem = NULL;
                        break;
                    }
                    else
                    {
                        // Continue to dequeue next item
                    }
                }
                else
                {
                    // Receive other item's side request
                    auto it = mMap->find(sideSequ);
                    if (it != mMap->end())
                    {
                        it->second->Finish();
                    }
                }
            } // sideSequ > 0
            else
            {
                // This is an item response
                static_cast<ps::network::SeastarWorkItem*>(ptr)->Complete();
                ptr->Deprecate();
                if (ptr == mMyItem)
                {
                    if (ptr->IsDeprecatedAndFinished())
                    {
                        mMap->erase(mMyItem->GetSequence());
                        delete mMyItem;
                        mMyItem = NULL;
                        break;
                    }
                    else
                    {
                        // Continue to dequeue next item
                    }
                }
                else
                {
                    // Continue to dequeue next item
                }
            }
        } // while (true)
    }
private:
    moodycamel::ReaderWriterQueue<T*>* mQueue;
    T* mMyItem;
    moodycamel::ReaderWriterQueue<T*>* mSendQ;
    uint64_t mSequence;
    unordered_map<uint64_t, T*>* mMap;
};

}
}
#endif // ML_PS5_NETWORK_QUEUE_HUB_FUTURE_H

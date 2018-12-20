#ifndef ML_PS5_NETWORK_QUEUE_HUB_H
#define ML_PS5_NETWORK_QUEUE_HUB_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <pthread.h>
#include "core/ps_queue_hub/readerwriterqueue.hh"
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/queue_hub_future.hh"
#include "core/ps_common.hh"

namespace ps
{
namespace network
{
#define NOTIFY_EPOLL_PWAIT_SIG SIGRTMIN+11
using namespace std;

class QueueHubFactory;  // Forward Declaration
template<typename T>
class QueueHub
{
public:
    QueueHub(int32_t mCount, int32_t nCount)
        : kMCount(mCount), kNCount(nCount), mQueueMatrix(), mSequence(1), mAssociatedHub(NULL)
    {
        // resize will init mTargetSeastarThreadIds[i]=false
        // init mNeedNotifyEpolls[i]=0
        mTargetSeastarThreadIds.resize(nCount);
        mNeedNotifyEpolls.resize(nCount);

        mQueueMatrix.resize(kMCount);
        mMapMatrix.resize(kMCount);
        for (int32_t i = 0; i < kMCount; ++i)
        {
            mQueueMatrix[i].resize(kNCount);
            mMapMatrix[i].resize(kNCount);
            for (int32_t j = 0; j < kNCount; ++j)
            {
                mQueueMatrix[i][j] = new moodycamel::ReaderWriterQueue<T*>(16384);
            }
        }
    }
    ~QueueHub()
    {
        for (int32_t i = 0; i < kMCount; ++i)
        {
            for (int32_t j = 0; j < kNCount; ++j)
            {
                delete mQueueMatrix[i][j];
            }
        }
    }
    void EnqueueBack(T* input, int32_t sour, int32_t dest)
    {
        while(!mQueueMatrix[sour][dest]->try_enqueue(input));
    }
    ps::network::Future<T> Enqueue(T* input, int32_t sour, int32_t dest,
        ps::service::seastar::Closure* closure, bool shouldInsertIntoMap = true)
    {
        // wake up epoll if need
        if (mNeedNotifyEpolls[dest] && mTargetSeastarThreadIds[dest] > 0)
        {
            pthread_kill(mTargetSeastarThreadIds[dest], NOTIFY_EPOLL_PWAIT_SIG);
            mNeedNotifyEpolls[dest] = false;
        }
        uint64_t curSeq = mSequence.fetch_add(1, std::memory_order_relaxed);;
        input->SetSequence(curSeq);
        if (closure != NULL)
        {
            input->SetClosure(closure);
        }
        if (shouldInsertIntoMap)
        {
            mMapMatrix[sour][dest][curSeq] = input;
        }

        while(!mQueueMatrix[sour][dest]->try_enqueue(input));

        return ps::network::Future<T>(
            mAssociatedHub->mQueueMatrix[dest][sour], input,
            mQueueMatrix[sour][dest], &mMapMatrix[sour][dest]);
    }
    T* Dequeue(int32_t sour, int32_t dest)
    {
        T* ptr = NULL;
        if (mQueueMatrix[sour][dest]->try_dequeue(ptr))
        {
            return ptr;
        }
        return NULL;
    }
    ps::network::Future<T> Enqueue(T* input, /*int32_t sour,*/int32_t dest)
    {
        int32_t sour = GetPortNumberOnThread();
        // deafult: has no callback here
        return Enqueue(input, sour, dest, NULL);
    }
    T* Dequeue(int32_t sour/*, int32_t dest*/)
    {
        int32_t dest = GetPortNumberOnThread();
        return Dequeue(sour, dest);
    }
    ps::network::Future<T> GetFuture(int32_t sour, int32_t dest)
    {
        return ps::network::Future<T>(mQueueMatrix[sour][dest], NULL,
                mAssociatedHub->mQueueMatrix[dest][sour], NULL);
    }
    int32_t GetMCount() const
    {
        return kMCount;
    }
    int32_t GetNCount() const
    {
        return kNCount;
    }
    void SetNeedNotifyEpoll(unsigned which, bool need)
    {
        mNeedNotifyEpolls[which] = need;
    }
    void SetTargetSeastarThreadId(unsigned which, pthread_t id)
    {
        mTargetSeastarThreadIds[which] = id;
    }
    pthread_t GetTargetSeastarThreadId(unsigned which)
    {
        return mTargetSeastarThreadIds[which];
    }
    static int32_t GetPortNumberOnThread()
    {
        static thread_local int32_t currentThreadPort = -1;
        if (currentThreadPort < 0)
        {
            // It may not promise the atomicity.
            //currentThreadPort = sThreadPort++;
            currentThreadPort = sThreadPort.fetch_add(1, std::memory_order_relaxed);
        }
        return currentThreadPort;
    }
private:
    friend class QueueHubFactory;
    void SetAssociatedMatrix(QueueHub<T>* h)
    {
        mAssociatedHub = h;
    }
private:
    int32_t kMCount;
    int32_t kNCount;
    vector<vector<moodycamel::ReaderWriterQueue<T*>*>> mQueueMatrix;
    vector<vector<unordered_map<uint64_t, T*>>> mMapMatrix;
    std::atomic<uint64_t> mSequence;
    QueueHub<T>* mAssociatedHub;
    // wake up epoll or not
    vector<bool> mNeedNotifyEpolls;
    // the seastar thread ids that should be notify
    vector<pthread_t> mTargetSeastarThreadIds;
    static std::atomic<int32_t> sThreadPort;
};

template<typename T> std::atomic<int32_t> ps::network::QueueHub<T>::sThreadPort(0);

class QueueHubFactory
{
public:
    ~QueueHubFactory() {}
    static QueueHubFactory& GetInstance()
    {
        static std::mutex m;
        static std::unique_ptr<QueueHubFactory> f;
        if (!f)
        {
            std::lock_guard<std::mutex> g(m);
            if (!f) { f.reset(new QueueHubFactory()); }
        }
        return *f;
    }
    template<typename T> std::pair<QueueHub<T>*, QueueHub<T>*>
        GetHub(const std::string& name, int32_t m = 0, int32_t n = 0)
    {
        std::lock_guard<std::mutex> g(mMutex);
        if (mHubMap.find(name) == mHubMap.end())
        {
            QueueHub<T> * q = new QueueHub<T>(m, n);
            QueueHub<T> * r = new QueueHub<T>(n, m);
            q->SetAssociatedMatrix(r);
            r->SetAssociatedMatrix(q);
            mHubMap[name] = std::pair<void *, void *>(reinterpret_cast<void *>(q), reinterpret_cast<void *>(r));
        }
        return std::pair<QueueHub<T>*, QueueHub<T>*>(
                reinterpret_cast<QueueHub<T>*>(mHubMap[name].first),
                reinterpret_cast<QueueHub<T>*>(mHubMap[name].second));
    }
    template<typename T> std::pair<QueueHub<T>*, QueueHub<T>*>
        GetHubWithoutLock(const std::string& name)
    {
        if (mHubMap.find(name) == mHubMap.end())
        {
            return std::pair<QueueHub<T>*, QueueHub<T>*>(
                    reinterpret_cast<QueueHub<T>*>(NULL),
                    reinterpret_cast<QueueHub<T>*>(NULL));
        }
        return std::pair<QueueHub<T>*, QueueHub<T>*>(
                reinterpret_cast<QueueHub<T>*>(mHubMap[name].first),
                reinterpret_cast<QueueHub<T>*>(mHubMap[name].second));
    }

    // The default function just for Item type
    // help function for client and server,
    // the default MxN queue name is "SEASTAR",
    // in server_client mode(server and client will be launched in one node), 
    // we will create an extra queue for server, named "SEASTAR2"
    std::pair<QueueHub<Item>*, QueueHub<Item>*> GetDefaultQueueForClient();
    std::pair<QueueHub<Item>*, QueueHub<Item>*> GetDefaultQueueForServer();

private:
    QueueHubFactory() : mMutex(), mHubMap() {}
    QueueHubFactory(const QueueHubFactory& other);
    QueueHubFactory& operator=(const QueueHubFactory& other);
    std::mutex mMutex;
    std::unordered_map<std::string, std::pair<void*, void*> > mHubMap;
};

}
}
#endif // ML_PS5_NETWORK_QUEUE_HUB_H

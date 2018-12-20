#ifndef ML_PS5_NETWORK_PS_QUEUE_WORK_ITEM_H
#define ML_PS5_NETWORK_PS_QUEUE_WORK_ITEM_H

#include <iostream>
#include <string>
#include <memory>
#include <exception>
#include <vector>
#include <mutex>
#include "core/future.hh"
#include "core/ps_queue_hub/queue_work_item.hh"

namespace ps
{
namespace network
{
using namespace std;
using Callback = std::function<void ()>;

class PsWorkItem : public ps::network::Item
{
public:
    PsWorkItem() : ps::network::Item(PsEngine)
    {
        mCallback = [] () {};
    }
    virtual ~PsWorkItem()
    {}
    virtual void Run()
    {
        //Dequeue side (ps engine) logic
        //std::cout << "PsWorkItem Run" << std::endl;
        mCallback();
    }
    virtual seastar::future<> Complete()
    {
        // Enqueue side (seastar) logic
        //std::cout << "PsWorkItem Complete" << std::endl;
        return seastar::make_ready_future<>();
    };
    virtual void SetCallback(Callback&& cb)
    {
        mCallback = std::move(cb);
    }
private:
    Callback mCallback;
};

} // namespace
} // namespace

#endif // ML_PS5_NETWORK_PS_QUEUE_WORK_ITEM_H

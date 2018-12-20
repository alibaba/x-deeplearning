/*
 * This file is open source software, licensed to you under the terms
 * of the Apache License, Version 2.0 (the "License").  See the NOTICE file
 * distributed with this work for additional information regarding copyright
 * ownership.  You may not use this file except in compliance with the License.
 *
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * Copyright (C) 2014 Cloudius Systems, Ltd.
 */

#include <memory>
#include "core/app-template.hh"
#include "core/reactor.hh"
#include "core/print.hh"
#include "core/ps_queue_hub/unittest/boost/threadpool.hpp"
#include "core/ps_queue_hub/queue_hub.hh"
#include "core/ps_queue_hub/queue_work_item.hh"
#include "core/ps_queue_hub/queue_hub_future.hh"
#include "core/ps_queue_hub/ps_work_item.hh"
#include "core/ps_coding/unittest/echo_request_item.hh"
#include <chrono>

using namespace std;
using namespace seastar;
using namespace std::chrono_literals;

#define BUG() do { \
        std::cerr << "ERROR @ " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("test failed"); \
    } while (0)

#define OK() do { \
        std::cerr << "OK @ " << __FILE__ << ":" << __LINE__ << std::endl; \
    } while (0)

template <typename Clock>
struct timer_test {
    timer<Clock> t1;
    timer<Clock> t2;
    timer<Clock> t3;
    timer<Clock> t4;
    timer<Clock> t5;
    promise<> pr1;
    promise<> pr2;

    future<> run() {
        t1.set_callback([this] {
            OK();
            print(" 500ms timer expired\n");
            if (!t4.cancel()) {
                BUG();
            }
            if (!t5.cancel()) {
                BUG();
            }
            t5.arm(1100ms);
        });
        t2.set_callback([] { OK(); print(" 900ms timer expired\n"); });
        t3.set_callback([] { OK(); print("1000ms timer expired\n"); });
        t4.set_callback([] { OK(); print("  BAD cancelled timer expired\n"); });
        t5.set_callback([this] { OK(); print("1600ms rearmed timer expired\n"); pr1.set_value(); });

        t1.arm(500ms);
        t2.arm(900ms);
        t3.arm(1000ms);
        t4.arm(700ms);
        t5.arm(800ms);

        return pr1.get_future().then([this] { return test_timer_cancelling(); });
    }

    future<> test_timer_cancelling() {
        timer<Clock>& t1 = *new timer<Clock>();
        t1.set_callback([] { BUG(); });
        t1.arm(100ms);
        t1.cancel();

        t1.arm(100ms);
        t1.cancel();

        t1.set_callback([this] { OK(); pr2.set_value(); });
        t1.arm(100ms);
        return pr2.get_future().then([&t1] { delete &t1; });
    }
};

class Producer
{
public:
    Producer(int32_t id) : mProducerId(id)
    {
        std::pair<ps::network::QueueHub<ps::network::Item>*,
            ps::network::QueueHub<ps::network::Item>*> mQueueHubPair
                = ps::network::QueueHubFactory::GetInstance().GetHub<ps::network::Item>("SEASTAR", 1, 2);
        mInputHub = mQueueHubPair.first;
        mOutputHub = mQueueHubPair.second;
    }
    void Produce()
    {
        /*
        ps::coding::unittest::EchoRequestItem* myItem0 = new ps::coding::unittest::EchoRequestItem(0);
        myItem0->SetReceiverId(0);
        ps::network::Future<ps::network::Item> f0 = mInputHub->Enqueue(myItem0, 0, 0);
        f0.Get();
        ps::network::SeastarWorkItem* myItem0 = new ps::network::SeastarWorkItem;
        ps::network::Future<ps::network::Item> f0 = mInputHub->Enqueue(myItem0, 0, 0);
        ps::network::SeastarWorkItem* myItem1 = new ps::network::SeastarWorkItem;
        ps::network::Future<ps::network::Item> f1 = mInputHub->Enqueue(myItem1, 0, 0);
        ps::network::SeastarWorkItem* myItem2 = new ps::network::SeastarWorkItem;
        ps::network::Future<ps::network::Item> f2 = mInputHub->Enqueue(myItem2, 0, 1);
        ps::network::SeastarWorkItem* myItem3 = new ps::network::SeastarWorkItem;
        ps::network::Future<ps::network::Item> f3 = mInputHub->Enqueue(myItem3, 0, 1);
        f0.Get();
        std::cout << "Produce: f0.Get() finished" << std::endl;
        f1.Get();
        std::cout << "Produce: f1.Get() finished" << std::endl;
        f2.Get();
        std::cout << "Produce: f2.Get() finished" << std::endl;
        f3.Get();
        std::cout << "Produce: f3.Get() finished" << std::endl;
        */
    }
private:
    int32_t mProducerId;
    std::pair<ps::network::QueueHub<ps::network::Item>*,
        ps::network::QueueHub<ps::network::Item>*> mQueueHubPair;
    ps::network::QueueHub<ps::network::Item>* mInputHub;
    ps::network::QueueHub<ps::network::Item>* mOutputHub;
};

int main(int ac, char** av) {
    static const int32_t kClientCount = 1;
    app_template app;
    timer_test<steady_clock_type> t1;
    timer_test<lowres_clock> t2;
    boost::threadpool::pool producerThreadPool(kClientCount);
    std::vector<Producer*> producers;
    for (int32_t i = 0; i < kClientCount; ++i)
    {
        producers.push_back(new Producer(i));
        producerThreadPool.schedule(boost::bind(&Producer::Produce, producers[i]));
    }
    return app.run_deprecated(ac, av, [&t1, &t2] {
        print("=== Start High res clock test\n");
        t1.run().then([&t2] {
            print("=== Start Low  res clock test\n");
            return t2.run();
        }).then([] {
            print("Done\n");
            engine().exit(0);
        });
    });
}

/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "gtest/gtest.h"
#include "xdl/core/ops/ps_ops/model_server/model_server.h"
#include "xdl/core/ops/ps_ops/model_server/model_server_forward.h"
#include "xdl/core/ops/ps_ops/model_server/model_server_backward.h"

using xdl::ModelServer;
using xdl::PsModelServerForwardItem;
using xdl::PsModelServerForwardQueue;
using xdl::PsModelServerBackwardItem;
using xdl::PsModelServerBackwardQueue;

TEST(ModelServerTest, ModelServer) {
  {
    auto ms = new ModelServer("schedule_addr", 0, 1, "name=no_cache", "name=no_cache");
    ASSERT_NE(ms, nullptr);
    xdl::Status st = ms->Init();
    ASSERT_EQ(st, xdl::Status::Ok());
    ASSERT_EQ(ms->ForwardHandle(), 0);
    ASSERT_EQ(ms->BackwardHandle(), 0);
    delete ms;
  }

  {
    auto psfi = new PsModelServerForwardItem();
    ASSERT_NE(psfi, nullptr);
    psfi->run = false;
    delete psfi;
  }

  {
    auto queue = PsModelServerForwardQueue::Get();
    int ret = queue->NewHandle();
    ASSERT_EQ(ret, 1);

    auto cq = queue->Queue(0);
    ASSERT_NE(cq, nullptr);

    auto waiter = [](void) {};
    queue->Wait(0, waiter);

    std::unique_ptr<PsModelServerForwardItem> item(new PsModelServerForwardItem());
    auto cb = [](PsModelServerForwardItem* item) {ASSERT_NE(item, nullptr);};
    queue->Push(0, item.get());

    queue->Pop(0, cb);
  }

  {
    auto queue = PsModelServerBackwardQueue::Get();
    int ret = queue->NewHandle();
    ASSERT_EQ(ret, 1);

    auto cq = queue->Queue(0);
    ASSERT_NE(cq, nullptr);

    auto waiter = [](void) {};
    queue->Wait(0, waiter);

    std::unique_ptr<PsModelServerBackwardItem> item(new PsModelServerBackwardItem());
    auto cb = [](PsModelServerBackwardItem* item) {ASSERT_NE(item, nullptr);};
    queue->Push(0, item.get());

    queue->Pop(0, cb);
  }

}

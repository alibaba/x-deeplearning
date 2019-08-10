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
#include "ps-plus/message/streaming_model_infos.h"
#include "ps-plus/message/streaming_model_manager.h"

using ps::StreamingModelManager;
using ps::StreamingModelWriter;
using ps::DenseVarNames;
using ps::DenseVarValues;
using ps::Tensor;
using ps::Status;

TEST(StreamingTest, Streaming) {
  {
    auto dvn = new DenseVarNames();
    dvn->names.push_back("hello");
    dvn->names.push_back("world");
    ASSERT_EQ(dvn->names.size(), 2);
    delete dvn;
  }

  {
    auto dvv = new DenseVarValues();
    struct DenseVarValues::DenseVarValue vv;
    vv.name = "robin";
    vv.offset = 0;
    Tensor tensor;
    vv.data = tensor;
    dvv->values.push_back(vv);
    ASSERT_EQ(dvv->values.size(), 1);
    delete dvv;
  }

  {
    Tensor tensor;

    auto dm = new StreamingModelWriter::DenseModel();
    dm->name = "first";
    dm->data = tensor;
    ASSERT_NE(dm, nullptr);
    delete dm;

    auto sm = new StreamingModelWriter::SparseModel();
    sm->name = "second";
    sm->data = tensor;
    sm->ids.push_back(1);
    sm->ids.push_back(2);
    sm->offsets.push_back(3);
    sm->offsets.push_back(4);
    ASSERT_EQ(sm->ids.size(), 2);
    delete sm;

    auto hm = new StreamingModelWriter::HashModel();
    hm->ids.push_back(std::make_pair(3, 9));
    hm->ids.push_back(std::make_pair(9, 81));
    ASSERT_EQ(hm->ids.size(), 2);
    delete hm;
  }
}

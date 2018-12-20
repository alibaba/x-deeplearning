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

#include "xdl/data_io/batch.h"
#include "gtest/gtest.h"

namespace xdl {
namespace io {

class MyBatch : public Batch {
public:
  size_t GetBlockSize(void) {
    std::map<std::string, Block> blocks = this->blocks();
    return blocks.size();
  }
};

TEST(BatchTest, Batch) {
  {
    std::unique_ptr<Batch> ba(new Batch());
    EXPECT_NE(ba, nullptr);

    Block block;
    block.ts_[Block::kIndex] = new Tensor();

    bool res = ba->Add("hello", block);
    EXPECT_EQ(res, true);

    auto tensor = ba->GetTensor("hello", Block::kIndex);
    EXPECT_NE(tensor, nullptr);
  }

  {
    std::unique_ptr<MyBatch> mb(new MyBatch());
    EXPECT_NE(mb, nullptr);

    size_t ret = mb->GetBlockSize();
    EXPECT_EQ(ret, 0);
  }
}

}
}

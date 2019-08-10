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

#include "xdl/data_io/merger/merger.h"
#include "xdl/core/lib/unique.h"

#include "xdl/data_io/constant.h"
#include "xdl/core/framework/cpu_device.h"
#include "xdl/core/lib/timer.h"
#include "xdl/core/utils/logging.h"

//#include "xdl/core/framework/gpu/gpu_device.h"

namespace xdl {
namespace io {

Merger::Merger(const Schema *schema, Device *dev)
    : schema_(schema), dev_(dev) {
}

bool Merger::Init() {
  return true;
}

Batch *Merger::Run(Batch *batch) {
  //XDL_TIMER_SCOPE(merger_run);
  auto &blocks = batch->blocks();
  XDL_CHECK(blocks.size() != 0);
  for (auto &kv : blocks) {
    auto &name = kv.first;
    auto &blk = kv.second;
    if (blk.ts_[Block::kKey] == nullptr) {
      continue;
    }
    if (blk.ts_[Block::kIndex] == nullptr) {
      blk.ts_[Block::kIndex] = new Tensor();
    }
    if (blk.ts_[Block::kUKey] == nullptr) {
      blk.ts_[Block::kUKey] = new Tensor();
    }
    if (blk.ts_[Block::kSIndex] == nullptr) {
      blk.ts_[Block::kSIndex] = new Tensor();
    }
    if (blk.ts_[Block::kSSegment] == nullptr) {
      blk.ts_[Block::kSSegment] = new Tensor();
    }
    auto fn = functor::UniqueFunctor<CpuDevice, int64_t, int32_t>();
    fn((CpuDevice *)dev_, *blk.ts_[Block::kKey], *blk.ts_[Block::kSegment], blk.ts_[Block::kUKey], blk.ts_[Block::kIndex], blk.ts_[Block::kSIndex], blk.ts_[Block::kSSegment]);
  }
  return batch;
}

}  // namespace io
}  // namespace xdl

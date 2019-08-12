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

#include "ps-plus/client/partitioner/merged_broadcast.h"

namespace ps {
namespace client {
namespace partitioner {

Status MergedBroadcast::Split(MergedPartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
  size_t part_size = ctx->GetContext(0)->GetVariableInfo()->parts.size();
  for (size_t i = 0; i < ctx->ContextSize(); ++i) {
    if (ctx->GetContext(i)->GetVariableInfo()->parts.size() != part_size) {
      return Status::ArgumentError("Merged Hash Variable Should Have the Same Parts Size");
    }
  }
  dst->clear();
  for (size_t i = 0; i < part_size; ++i) {
    dst->push_back(src);
  }
  return Status::Ok();
}

}
}
}


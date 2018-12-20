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

#include "ps-plus/client/partitioner/logic.h"
#include <cstring>
#include <iostream>

namespace ps {
namespace client {
namespace partitioner {

Status Logic::Combine(PartitionerContext* ctx, Data* src, size_t server_id, std::unique_ptr<Data>* output) {
  WrapperData<bool>* raw_src = dynamic_cast<WrapperData<bool>*>(src);
  if (raw_src == nullptr) {
    return Status::ArgumentError("Logic Partitioner Combine src should be bool");
  }

  if (output->get() == nullptr) {
    output->reset(new WrapperData<bool>(true));
  } 

  WrapperData<bool>* dst = dynamic_cast<WrapperData<bool>*>(output->get());
  if (!raw_src->Internal() && dst->Internal()) {
    dst->Internal() = false;
  }

  return Status::Ok();
}

}
}
}


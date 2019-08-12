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

#include "xdl/data_io/parser/parse_pb.h"

#include <assert.h>

#include "xdl/data_io/pool.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

ssize_t ParsePB::GetSize(const char *str, size_t len) {
  size_t size = ((uint32_t *)str)[0];
  size += sizeof(uint32_t);
  if (size > len) {
    return -1;
  }
  return size;
}

SGroup *ParsePB::Run(const char *str, size_t len) {
  if (str == nullptr || len == 0) {
    return nullptr;
  }
  SGroup *sgroup = SGroupPool::Get()->Acquire();
  SampleGroup *sg = sgroup->New();
  XDL_CHECK(sg->ParseFromArray(str + sizeof(uint32_t), len - sizeof(uint32_t)))
      << "parse sample group failed, len=" << len;
  sgroup->Reset();
  return sgroup;
}

}  // namespace xdl
}  // namespace io

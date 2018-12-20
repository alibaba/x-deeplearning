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

/*
 * Copyright 1999-2018 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef XDL_IO_BATCH_H_
#define XDL_IO_BATCH_H_

#include "xdl/data_io/sgroup.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/utils/logging.h"

#include <atomic>

namespace xdl {
namespace io {

static const char *kSKeyName = "skey";
static const char *kLabelName = "label";
static const char *kIndicatorPrefix = "ind.";

struct Block {
  enum Type {
    kValue = 0,      // float dense, sparse, label
    kIndex,          // int32/int64 sample key index, indicator, uniq index
    kSegment,        // int32 sparse
    kKey,            // int64 key, sparse key
    kUKey,           // int64 key, unique sparse key
    kSBuf,           // int8 sample key
    kTypes,          // number of type
  };
  Tensor *ts_[kTypes];
  size_t ts_count_ = 0;
  bool valid_;
};

class Batch {
 public:
  Block *GetMutable(const std::string &name);
  const Block *Get(const std::string &name) const;
  const Tensor *GetTensor(const std::string &name, Block::Type type) const;

  bool Add(const std::string &name, const Block &block);
  bool Keep(SGroup *sgroup);
  bool Reuse();
  std::vector<SGroup *> &sgroups();
  std::map<std::string, Block> &blocks();
  std::atomic<size_t> ts_count_;
 protected:
  std::map<std::string, Block> blocks_;
  std::vector<SGroup *> sgroups_;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_BATCH_H_

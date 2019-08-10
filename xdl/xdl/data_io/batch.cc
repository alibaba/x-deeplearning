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
#include "xdl/data_io/pool.h"

#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

Block *Batch::GetMutable(const std::string &name) {
  return &blocks_[name];
}

const Block *Batch::Get(const std::string &name) const {
  auto const it = blocks_.find(name);
  return it == blocks_.end() ? nullptr : &(it->second);
}

const Tensor *Batch::GetTensor(const std::string &name, Block::Type type) const {
  XDL_CHECK(type < Block::kTypes);
  auto it = blocks_.find(name);
  return it == blocks_.end() ? nullptr : it->second.ts_[type];
}

bool Batch::Add(const std::string &name, const Block &block) {
  blocks_.insert(std::make_pair(name, block));
  return true;
}

bool Batch::Keep(SGroup *sgroup) {
  sgroups_.push_back(sgroup);
  return true;
}

bool Batch::Reuse() {
  for (int i = 0; i < sgroups_.size(); ++i) {
    auto sgroup = sgroups_.at(i);
    sgroup->Reuse();
  }
  sgroups_.clear();
  ts_count_ = 0;
  abandon_ = false;
  BatchPool::Get()->Release(this);
  XDL_LOG(DEBUG) << "release batch=" << this;
  return true;
}

std::vector<SGroup *> &Batch::sgroups() {
  return sgroups_;
}

std::map<std::string, Block> &Batch::blocks() {
  return blocks_;
}

}  // namespace io
}  // namespace xdl

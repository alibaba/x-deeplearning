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

#include "xdl/data_io/sgroup.h"
#include "xdl/data_io/pool.h"

#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

SGroup::SGroup() {
}

SGroup::SGroup(const SGroup &sgroup) {
  if (sg_ != nullptr && own_) {
    delete sg_;
  }
  own_ = false;
  sg_ = sgroup.sg_;
  size_ = sgroup.size_;
  begin_ = sgroup.begin_;
  end_ = sgroup.end_;
}

void SGroup::CloneTail(SGroup *sgroup, int end) {
  if (sg_ != nullptr && own_) {
    delete sg_;
  }
  *this = *sgroup;
  sgroup->own_ = false;
  Reset(sgroup->end_, end==0?sgroup->size_:end);
}

void SGroup::Reset(int begin, int end) {
  XDL_CHECK(begin >= 0);
  begin_ = begin;
  if (end == 0) {
    XDL_CHECK(sg_->labels_size() > 0);
    end_ = size_ = sg_->labels_size();
    XDL_CHECK(size_ > 0);
  } else {
    XDL_CHECK(end > begin_ && end <= size_);
    end_ = end;
  }
}

bool SGroup::Reuse() {
  SGroupPool::Get()->Release(this);
  //XDL_LOG(DEBUG) << "release sgroup=" << this;
  return true;
}

SampleGroup *SGroup::New() {
  if (sg_ != nullptr && own_) {
    delete sg_;
  }
  sg_ = new SampleGroup();
  own_ = true;
  return sg_;
}

SampleGroup *SGroup::Get() {
  XDL_CHECK(sg_ != nullptr);
  return sg_;
}

const SampleGroup *SGroup::Get() const {
  XDL_CHECK(sg_ != nullptr);
  return sg_;
}

const bool SGroup::empty() const {
  XDL_CHECK(sg_ != nullptr);
  return size_ == 0;
}


}  // namespace io
}  // namespace xdl

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

#ifndef XDL_IO_DATA_SGROUP_H_
#define XDL_IO_DATA_SGROUP_H_

#include "xdl/proto/sample.pb.h"

namespace xdl {
namespace io {

class SGroup {
 public:
  SGroup();
  SampleGroup *Get();
  const SampleGroup *Get() const;
  SampleGroup *New();
  const bool empty() const;

  void Reset(int begin=0, int end=0);
  void CloneTail(SGroup *sg, int end=0);

  bool Reuse();

  int size_ = 0;
  int begin_ = 0;
  int end_ = 0;
  int own_ = false;
 private:
  SampleGroup *sg_;
  SGroup(const SGroup &sgroup);
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_DATA_SGROUP_H_
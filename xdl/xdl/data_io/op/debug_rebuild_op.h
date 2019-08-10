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


#ifndef XDL_CORE_IO_DEBUG_REBUILD_OP_H_
#define XDL_CORE_IO_DEBUG_REBUILD_OP_H_

#include "xdl/data_io/op/op.h"

#include <mutex>

namespace xdl {
namespace io {

class DebugRebuildOP : public Operator {
 public:
  DebugRebuildOP() {}
  virtual ~DebugRebuildOP() {}

  virtual bool Init(const std::map<std::string, std::string> &params) override;
  virtual bool Run(SampleGroup *sample_group) override;

 private:
  std::mutex mutex_;
  size_t repeats_ = 0;
  size_t limit_ = 4;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_DEMO_H_
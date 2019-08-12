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


#ifndef XDL_CORE_IO_PACK_LABEL_H_
#define XDL_CORE_IO_PACK_LABEL_H_

#include "xdl/data_io/packer/packer.h"

namespace xdl {
namespace io {

class PackLabel : public Pack {
 public:
  PackLabel(Device *dev, const Schema *schema) : Pack(dev, schema) {}
  virtual ~PackLabel() {}

  /// 1. init each new batch
  virtual bool Init(Batch *batch) override;

  /// 2. run each sgroup
  virtual std::pair<int, int> Stat(const PParam &pparam) override;

  /// 3. setup for batch
  virtual bool Setup() override;

  /// 4. run each sgroup
  virtual std::pair<int, int> Run(const PParam &pparam) override;

 protected:
  size_t n_ = 0;
  size_t label_count_ = 0;
  off_t offset_ = 0;
  Block *blk_ = nullptr;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_PACK_LABEL_H_
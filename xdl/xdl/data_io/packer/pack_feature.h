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

#ifndef XDL_CORE_IO_PACK_FEATURE_H_
#define XDL_CORE_IO_PACK_FEATURE_H_

#include "xdl/data_io/packer/packer.h"

#include <map>

namespace xdl {
namespace io {

class PackFeature : public Pack {
 public:
  PackFeature(Device *dev, const Schema *schema) : Pack(dev, schema) {}
  virtual ~PackFeature() {}

  /// 1. init each new batch
  virtual bool Init(Batch *batch) override;

  /// 2. run each sgroup
  virtual std::pair<int, int> Stat(const PParam &pparam) override;

  /// 3. setup for batch
  virtual bool Setup() override;

  /// 4. run each sgroup
  virtual std::pair<int, int> Run(const PParam &pparam) override;

 protected:
  struct FStat {
    inline void Reset() {
      n_ = 0;
      off_ = 0;
      blk_ = nullptr;
    }
    size_t w_ = 0;     // nvec
    size_t n_ = 0;     // feature count of this batch
    off_t off_ = 0;    // write offset of this batch
    size_t seq_ = 0;   // seq point to tables_seq_[ktable]
    const FeatureOption *opt_ = nullptr;
    Block *blk_ = nullptr;
  };

  bool InitFStats();
  inline size_t TableN(const FStat &stat) const;
  inline size_t TableN(size_t ktable) const;
  inline size_t BatchSize(const FStat &stat) const;
  inline size_t BatchSize(size_t ktable) const;

  size_t tables_n_[kTablesMax];
  size_t tables_off_[kTablesMax];
  std::vector<FStat *> tables_seq_[kTablesMax];
  std::map<std::string, FStat> feature_stats_;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_PACK_FEATURE_H_

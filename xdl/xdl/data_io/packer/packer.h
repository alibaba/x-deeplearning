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


#ifndef XDL_CORE_IO_PACKER_H_
#define XDL_CORE_IO_PACKER_H_

#include "xdl/data_io/sgroup.h"
#include "xdl/data_io/batch.h"
#include "xdl/data_io/schema.h"
#include "xdl/core/framework/tensor.h"

namespace xdl {
namespace io {

struct PParam {
  const ::google::protobuf::RepeatedPtrField<std::string> *sample_ids_;
  const ::google::protobuf::RepeatedPtrField<Label> *labels_;
  const FeatureTable *ftable_;
  int ktable_;
  int ntable_;
  int isgroup_;
  int begin_;
  int end_;
};

class Pack {
 public:
  Pack(Device *dev, const Schema *schema): dev_(dev), schema_(schema) {}
  virtual ~Pack() {}

  /// 1. init each new batch
  virtual bool Init(Batch *batch) {
    batch_ = batch;
  }

  /// 2. run each sgroup
  virtual std::pair<int, int> Stat(const PParam &pparam) = 0;

  /// 3. setup for batch
  virtual bool Setup() = 0;

  /// 4. run each sgroup
  virtual std::pair<int, int> Run(const PParam &pparam) = 0;

 protected:
  Device *dev_ = nullptr;
  const Schema *schema_ = nullptr;
  Batch *batch_ = nullptr;
};

class Packer {
 public:
  Packer(const Schema *schema, Device *dev);
  virtual ~Packer() {}

  bool Init();

  /* assume total size of sgroups less than or equal to batch_size */
  Batch *Run(std::vector<SGroup*> *sgroups, size_t *total_size_p);

  /* return batch, nullptr, or END */
  std::vector<Batch *>Run(SGroup *sgroup);

 protected:
  Device *dev_ = nullptr;
  const Schema *schema_ = nullptr;

  enum PackDef {
    kPackSKey = 0,
    kPackLabel,
    kPackFeature,
    kPackCount,
  };

  Pack *packs_[kPackCount];
  std::vector<SGroup*> sgroups_;
  size_t total_size_ = 0;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_PACKER_H_
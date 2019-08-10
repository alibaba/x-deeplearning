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

#ifndef XDL_CORE_IO_PACK_FEATURE_H_
#define XDL_CORE_IO_PACK_FEATURE_H_

#include "xdl/data_io/packer/packer.h"
#include "xdl/core/utils/logging.h"

#include <unordered_map>
#include <bitset>

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
  /// stat
  struct Stat {
    inline void Reset() {
      n_ = 0;
      off_ = 0;
      blk_ = nullptr;
    }
    size_t n_ = 0;                        // count of feature value in feature, group size, table size, counting while <Stat>
    off_t off_ = 0;                       // inner offset of state, updating while <Run>
    Block *blk_ = nullptr;                // block for feature, coordinates, indicator, updating while <Run>
  };

  /// feature stat
  struct FStat : public Stat {
    inline void Reset() {
      Stat::Reset();
    }
    const FeatureOption *opt_ = nullptr;  // feature option
    size_t w_ = 0;                        // nvec
    size_t seq_ = 0;                      // seq point to table_stat.seq_
    std::bitset<kNVecMax> mask_;          // mask for dense
  };
  
  /// table stat
  struct TStat : public Stat {
    inline void Reset() {
      Stat::Reset();
    }
    std::vector<FStat *> seq_;            // feature seq of this table
    unsigned k_ = 0;                      // ktable
  };

  inline size_t TableN(const FStat &stat) const;
  inline size_t TableN(size_t ktable) const;
  inline size_t TableSize(const FStat &stat) const;
  inline size_t TableSize(size_t ktable) const;
  inline size_t GroupSize() const;

  bool InitStats();
  /// return value count
  size_t OnFeature(FStat *stat, off_t offset, const Feature &f, int cutoff);
  bool FeaturePad(FStat *stat, size_t offset, size_t end);
  bool OnIndicator(TStat *stat, const FeatureLine &fl);
  bool IndicatorPad(TStat *stat, off_t end);

  //size_t tables_n_[kTablesMax];
  //size_t tables_off_[kTablesMax];
  std::vector<FStat *> tables_seq_[kTablesMax];

  std::vector<TStat> table_stats_;
  std::unordered_map<std::string, FStat> feature_stats_;
};

inline size_t PackFeature::TableN(const FStat &stat) const {
  if (stat.opt_->table() >= table_stats_.size()) {
    return 0;
  }
  return table_stats_[stat.opt_->table()].n_;
}

inline size_t PackFeature::TableN(size_t ktable) const {
  if (ktable >= table_stats_.size()) {
    return 0;
  }
  return table_stats_[ktable].n_;
}

inline size_t PackFeature::TableSize(const FStat &stat) const {
  return TableSize(stat.opt_->table());
}

inline size_t PackFeature::TableSize(size_t ktable) const {
  if (ktable == 0 && schema_->padding_) {
    return schema_->batch_size_;
  }
  XDL_CHECK(ktable < table_stats_.size());
  size_t bs = table_stats_[ktable].n_;
  if (ktable > 0 && table_stats_[0].n_ < schema_->batch_size_ && table_stats_[ktable].n_ < schema_->batch_size_ && schema_->padding_) {
    XDL_DLOG(DEBUG) << "extend a zero line for padding, ktable=" << ktable
        << " table.n=" << table_stats_[ktable].n_;
    bs += 1;
  }
  return bs;
}

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_PACK_FEATURE_H_

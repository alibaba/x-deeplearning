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
#include "xdl/data_io/packer/pack_feature.h"

namespace xdl {
namespace io {

bool PackFeature::Init(Batch *batch) {
  Pack::Init(batch);

  memset(tables_n_, 0, sizeof(tables_n_));
  memset(tables_off_, 0, sizeof(tables_off_));

  InitFStats();

  return true;
}

bool PackFeature::InitFStats() {
  for (auto &it : feature_stats_) {
    auto &stat = it.second;
    stat.Reset();
  } 
  if (feature_stats_.size() != 0) {
    return true;
  }

  auto &opts = schema_->feature_opts();
  for (auto &it: opts) {
    auto &opt = it.second;
    //std::cout << opt->name() << std::endl;
    auto &stat = feature_stats_[opt->name()];
    stat.opt_ = opt;
    stat.seq_ = tables_seq_[opt->table()].size();
    size_t nvec = stat.opt_->has_nvec() ? stat.opt_->nvec() : 1;
    stat.w_ = nvec;
    XDL_CHECK(stat.w_ > 0);
    tables_seq_[opt->table()].push_back(&stat);
  }

  XDL_CHECK(feature_stats_.size() != 0);

  return true;
}

inline size_t PackFeature::TableN(const FStat &stat) const {
  return tables_n_[stat.opt_->table()];
}

inline size_t PackFeature::TableN(size_t ktable) const {
  return tables_n_[ktable];
}

inline size_t PackFeature::BatchSize(const FStat &stat) const {
  return BatchSize(stat.opt_->table());
}

inline size_t PackFeature::BatchSize(size_t ktable) const {
  if (ktable == 0 && schema_->padding_) {
    return schema_->batch_size_;
  }
  size_t bs = tables_n_[ktable];
  if (ktable > 0 && tables_n_[0] < schema_->batch_size_ && tables_n_[ktable] < schema_->batch_size_ && schema_->padding_) {
    XDL_DLOG(DEBUG) << "extend a zero line, ktable=" << ktable
        << " table_n=" << tables_n_[ktable];
    bs += 1;
  }
  return bs;
}

std::pair<int, int> PackFeature::Stat(const PParam &pparam) {
  XDL_CHECK(pparam.ftable_ != nullptr);

  auto ktable = pparam.ktable_;
  auto ftable = pparam.ftable_;
  XDL_CHECK(pparam.end_ <= ftable->feature_lines_size());


  int ref_l = INT_MAX, ref_r = INT_MIN;
  int begin = std::max(pparam.begin_, 0);
  int end = std::min(pparam.end_, ftable->feature_lines_size());

  //XDL_DLOG(DEBUG) << "stat[" << pparam.isgroup_ << ", " << ktable <<  "] (0)" << pparam.begin_ <<
  //    " -> " <<  pparam.end_ << "(" << ftable->feature_lines_size() << ")";

  for (int n = begin; n < end; ++n) {
    auto &fl = ftable->feature_lines(n);
    if (fl.has_refer()) {
      ref_l = std::min(ref_l, fl.refer());
      ref_r = std::max(ref_r, fl.refer());
    }

    /// foreach feature
    for (int f = 0; f < fl.features_size(); ++f) {
      auto &fea = fl.features(f);
      auto it = feature_stats_.find(fea.name());
      if (it == feature_stats_.end()) {
        continue;
      }

      auto &stat = it->second;
      XDL_CHECK(stat.opt_->table() == ktable) << "feature=" << fea.name()
          << " opt=" << stat.opt_
          << " opt.table=" << stat.opt_->table() << " ktable=" << ktable;

      if (stat.opt_->type() == kDense) {
        XDL_CHECK(fea.values_size() == 1) << "dense feature=" << stat.opt_->name() 
            << " must be presented as vector of one value, value size=" << fea.values_size();
      }
      stat.n_ += fea.values_size();
    }

    ++tables_n_[ktable];
    XDL_CHECK(tables_n_[ktable] <= schema_->batch_size_)
        << "table[" << ktable << "]" << " n=" << tables_n_[ktable]
        << " batch_size=" << schema_->batch_size_;
  }

  //XDL_DLOG(DEBUG) << "stat[" << pparam.isgroup_ << ", " << ktable <<  "] " << ref_l <<
  //    " -> " <<  ref_r + 1;

  return {ref_l, ref_r+1};
}

bool PackFeature::Setup() {
  batch_->ts_count_ = 0;
  for (auto &it: feature_stats_) {
    const std::string &name = it.first;
    auto &stat = it.second;
    auto ktable = stat.opt_->table();

    if (stat.blk_ == nullptr) {
      stat.blk_ = batch_->GetMutable(name);
      stat.blk_->ts_count_ = 0;
    }

    stat.blk_->valid_ = true;
    if (stat.opt_->type() == kSparse) {
      if (stat.blk_->ts_[Block::kValue] != nullptr) {
        delete stat.blk_->ts_[Block::kValue];
      }
      auto value = new Tensor(dev_, TensorShape({stat.n_, stat.w_}), types::kFloat);
      stat.blk_->ts_[Block::kValue] = value;

      if (stat.blk_->ts_[Block::kKey] != nullptr) {
        delete stat.blk_->ts_[Block::kKey];
      }

      auto key = new Tensor(dev_, stat.opt_->serialized()
                              ?TensorShape({stat.n_}):TensorShape({stat.n_, 2}), types::kInt64);
      stat.blk_->ts_[Block::kKey] = key;

      if (stat.blk_->ts_[Block::kSegment] != nullptr) {
        delete stat.blk_->ts_[Block::kSegment];
      }
      auto segment = new Tensor(dev_, TensorShape({BatchSize(stat)}), types::kInt32);
      stat.blk_->ts_[Block::kSegment] = segment;

      stat.blk_->ts_count_ = 3;
      batch_->ts_count_ += stat.blk_->ts_count_;
      //XDL_DLOG(DEBUG) << "create " << name << ".val " << value->Shape()
      //    << " " << name << ".idx " << key->Shape()
      //    << " " << name << ".seg " << segment->Shape();
    } else {
      if (stat.blk_->ts_[Block::kValue] != nullptr) {
        delete stat.blk_->ts_[Block::kValue];
      }
      auto value = new Tensor(dev_, TensorShape({BatchSize(stat), stat.w_}), types::kFloat);
      stat.blk_->ts_[Block::kValue] = value;
      stat.blk_->ts_count_ = 1;
      batch_->ts_count_ += stat.blk_->ts_count_;
      //XDL_DLOG(DEBUG) << "create " << name << ".val " << value->Shape();
    }
  }

  for (int k = 0; k < kTablesMax - 1 && TableN(k+1) > 0; ++k) {
    std::string name = kIndicatorPrefix + std::to_string(k);
    auto blk = batch_->GetMutable(name);
    if (blk->ts_[Block::kIndex] != nullptr) {
      delete blk->ts_[Block::kIndex];
    }
    auto indicator = new Tensor(dev_, TensorShape({BatchSize(k)}), types::kInt32);
    blk->ts_[Block::kIndex] = indicator;
    blk->ts_count_ = 1;
    batch_->ts_count_ += blk->ts_count_;
    //XDL_DLOG(DEBUG) << "create " << name << " " << indicator->Shape();
  }
  return true;
}

std::pair<int, int> PackFeature::Run(const PParam &pparam) {
  XDL_CHECK(pparam.ftable_ != nullptr && feature_stats_.size() != 0)
      << "ftable=" << pparam.ftable_ << " stats.size=" << feature_stats_.size();

  auto ktable = pparam.ktable_;
  auto ftable = pparam.ftable_;
  size_t batch_size = BatchSize(ktable);

  XDL_CHECK(pparam.end_ <= ftable->feature_lines_size());

  int ref_l = INT_MAX, ref_r = INT_MIN;
  int begin = std::max(pparam.begin_, 0);
  int end = std::min(pparam.end_, ftable->feature_lines_size());

  //XDL_DLOG(DEBUG) << "run[" << pparam.isgroup_ << ", " << ktable <<  "] (0)" << pparam.begin_ <<
  //    " -> " <<  pparam.end_ << "(" << ftable->feature_lines_size() << ")";

  for (int n = begin; n < end; ++n) {
    auto &fl = ftable->feature_lines(n);
    if (fl.has_refer()) {
      ref_l = std::min(ref_l, fl.refer());
      ref_r = std::max(ref_r, fl.refer());
    }

    std::vector<int8_t> feature_hits(tables_seq_[ktable].size());
    memset(&feature_hits[0], feature_hits.size(), 0);

    /// foreach feature
    for (int f = 0; f < fl.features_size(); ++f) {
      auto &fea = fl.features(f);
      auto it = feature_stats_.find(fea.name());
      if (it == feature_stats_.end()) {
        continue;
      }
      auto &stat = it->second;
      XDL_CHECK(stat.opt_->table() == ktable);

      //XDL_DLOG(DEBUG) << "feature["<< stat.seq_ << "]=" << stat.opt_->name() << " stat.off_=" << stat.off_;

      XDL_CHECK(stat.seq_ < feature_hits.size());
      feature_hits[stat.seq_] = 1;

      auto blk = stat.blk_;
      XDL_CHECK(blk != nullptr);

      /// foreach feature value
      bool sparse = stat.opt_->type() == kSparse;
      if (!sparse) {
        XDL_CHECK(fea.values_size() == 1);
      }

      for (int v = 0; v < fea.values_size(); ++v, ++stat.off_) {
        auto &val = fea.values(v);
        if (sparse) {
          XDL_CHECK(val.has_key());
          XDL_CHECK(blk->ts_[Block::kKey] != nullptr && blk->ts_[Block::kSegment] != nullptr);
          auto keys = blk->ts_[Block::kKey]->Raw<int64_t >();
          if (stat.opt_->serialized()) {
            keys[stat.off_] = val.key();
          } else {
            keys[stat.off_*2] = val.hkey();
            keys[stat.off_*2+1] = val.key();
          }
          sparse = true;
        } else {
          XDL_CHECK(stat.off_ == tables_off_[ktable]) << "feature=" << stat.opt_->name() <<
              " stat.off=" << stat.off_ << " != table[" << ktable << "].off=" << tables_off_[ktable];
        }

        if (val.has_value()) {
          XDL_CHECK(blk->ts_[Block::kValue] != nullptr);
          auto values = blk->ts_[Block::kValue]->Raw<float>();
          values[stat.off_] = val.value();
        } else if (val.vector_size() > 0) {
          XDL_CHECK(blk->ts_[Block::kValue] != nullptr);
          XDL_DCHECK(blk->ts_[Block::kValue]->Shape()[1] == val.vector_size()) 
              << "dense feature=" << stat.opt_->name() << " vector_size=" << val.vector_size()
              << " != " << " width=" << blk->ts_[Block::kValue]->Shape()[1];
          auto values = blk->ts_[Block::kValue]->Raw<float>();
          for (int m = 0; m < val.vector_size(); ++m) {
            values[stat.off_*val.vector_size()+m] = val.vector(m);
          }
        } else if (!val.has_value()) {
          XDL_CHECK(blk->ts_[Block::kValue] != nullptr);
          auto values = blk->ts_[Block::kValue]->Raw<float>();
          values[stat.off_] = 1.0;
        }
      }  // feature_value

      if (sparse) {
        auto segments = blk->ts_[Block::kSegment]->Raw<int32_t>();
        segments[tables_off_[ktable]] = stat.off_;
      }

    }  // feature

    // miss
    for (size_t i = 0; i < feature_hits.size(); ++i) {
      if (feature_hits[i] > 0) {
        //auto &stat = *(tables_seq_[ktable][i]);
        //XDL_DLOG(DEBUG) << "hit feature["<< stat.seq_ << "]=" << stat.opt_->name() << " stat.off_=" << stat.off_;
        continue;
      }
      auto &stat = *(tables_seq_[ktable][i]);

      //XDL_DLOG(DEBUG) << "missed feature["<< stat.seq_ << "]=" << stat.opt_->name() << " stat.off_=" << stat.off_;

      if (stat.opt_->type() == kSparse) {
        auto segment = stat.blk_->ts_[Block::kSegment];
        XDL_DCHECK(segment != nullptr && segment->Shape()[0] == batch_size);
        auto segments = segment->Raw<int32_t>();
        segments[tables_off_[ktable]] = stat.off_;
      } else {
        XDL_CHECK(stat.off_ == tables_off_[ktable]) << "feature=" << stat.opt_->name() <<
            " stat.off=" << stat.off_ << " != table[" << ktable << "].off=" << tables_off_[ktable];

        auto value = stat.blk_->ts_[Block::kValue];
        XDL_DCHECK(value != nullptr && value->Shape()[0] == batch_size && value->Shape()[1] == stat.w_);
        auto values = value->Raw<float>();
        for (int m = 0; m < stat.w_; ++m) {
          values[stat.off_*stat.w_+m] = 0;
        }
        ++stat.off_;
      }
    }

    if (fl.has_refer()) {
      XDL_CHECK(ktable < kTablesMax - 1 && TableN(ktable+1) > 0) 
          << "table[" << ktable << "].n=" << TableN(ktable) << " -> "
          << "table[" << ktable+1 << "].n=" << TableN(ktable+1);

      std::string name = kIndicatorPrefix + std::to_string(ktable);
      auto blk = batch_->GetMutable(name);
      auto indicator = blk->ts_[Block::kIndex];
      XDL_DCHECK(indicator != nullptr && indicator->Shape()[0]>= tables_n_[ktable])
          << "table[" << ktable << "].n=" << TableN(ktable) << " > "
          << name << "(" << indicator->Shape()[0] << ")";
      auto indicators = indicator->Raw<uint32_t>();
      indicators[tables_off_[ktable]] = tables_off_[ktable+1] + fl.refer(); 
    }

    ++tables_off_[ktable];
    //XDL_DLOG(DEBUG) << "table[" << ktable << "].off=" << tables_off_[ktable];

  }  /// for each feature_line

  /// padding main table & indicator
  if (tables_off_[ktable] == tables_n_[ktable] &&
      schema_->padding_ && tables_n_[ktable] < batch_size) {
    XDL_DLOG(DEBUG) << "batch finish, ktable=" << ktable 
        << " padding " << tables_n_[ktable] << " -> " << batch_size;

    for (auto &kv: tables_seq_[ktable]) {
      auto &stat = *kv;
      if (stat.opt_->type() == kSparse) {
        XDL_CHECK(stat.off_ == stat.n_) << "feature=" << stat.opt_->name()
            << " off=" << stat.off_ << " n=" << stat.n_;

        auto segment = stat.blk_->ts_[Block::kSegment];
        XDL_DCHECK(segment != nullptr && segment->Shape()[0] == batch_size)
            << "shape=(" << segment->Shape()[0] << ") batch_size=" << batch_size;
        auto segments = segment->Raw<int32_t>();
        for (int p = tables_off_[ktable]; p < batch_size; ++p) {
          segments[p] = 0;
          if (p > 0) {
            segments[p] += segments[p-1];
          }
        }
      } else {
        XDL_CHECK(stat.off_ == tables_off_[ktable]) << "feature=" << stat.opt_->name() <<
            " stat.off=" << stat.off_ << " != table[" << ktable << "].off=" << tables_off_[ktable];

        auto value = stat.blk_->ts_[Block::kValue];
        XDL_DCHECK(value != nullptr && value->Shape()[0] == batch_size && value->Shape()[1] == stat.w_);
        for (int p = tables_off_[ktable]; p < batch_size; ++p) {
          auto values = value->Raw<float>();
          for (int m = 0; m < stat.w_; ++m) {
            values[p*stat.w_+m] = 0;
          }
        }
      }

    }  /// for each stat in tables_seq_

    /// padding indicator to the padding line
    if (TableN(ktable + 1) > 0) {
      std::string name = kIndicatorPrefix + std::to_string(ktable);
      auto blk = batch_->GetMutable(name);
      auto indicator = blk->ts_[Block::kIndex];
      XDL_DCHECK(indicator != nullptr && indicator->Shape()[0] == batch_size)
          << indicator->Shape()[0] << " != " << batch_size;
      auto indicators = indicator->Raw<uint32_t>();
      int padding_refer = indicators[tables_off_[ktable]-1]+1;
      for (int i = tables_off_[ktable]; i < batch_size; ++i) {
        indicators[i] = indicators[tables_off_[ktable]-1];  // TODO: should be indicators[i] = padding_refer;
      }
    }
  }

  return {ref_l, ref_r+1};
}

}  // namespace io
}  // namespace xdl

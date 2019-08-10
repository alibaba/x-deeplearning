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
#include "xdl/data_io/packer/pack_feature.h"

namespace xdl {
namespace io {

bool PackFeature::Init(Batch *batch) {
  Pack::Init(batch);

  InitStats();

  return true;
}

bool PackFeature::InitStats() {
  for (auto &stat : table_stats_) {
    stat.Reset();
  } 
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
    if (table_stats_.size() < opt->table()+1) {
      table_stats_.resize(opt->table()+1);
    }

    auto &stat = feature_stats_[opt->name()];
    auto &tstat = table_stats_[opt->table()];
    stat.opt_ = opt;
    stat.seq_ = tstat.seq_.size();
    size_t nvec = stat.opt_->has_nvec() ? stat.opt_->nvec() : 1;
    stat.w_ = nvec;
    if (stat.opt_->has_mask()) {
      XDL_CHECK(stat.opt_->mask().size() <= nvec && nvec <= kNVecMax) << "mask length=" << stat.opt_->mask().size()
          << " nvec=" << nvec;
      std::string mask = stat.opt_->mask();
      //if (mask.size() < kNVecMax) {
      //  mask = std::string(kNVecMax-mask.size(), '1')+mask;
      //}
      XDL_LOG(DEBUG) << "mask: " << mask;
      stat.mask_ = std::bitset<kNVecMax>(stat.opt_->mask());
      XDL_CHECK(stat.mask_.any() && stat.mask_.count() <= nvec) << stat.mask_.count();
      stat.w_ = stat.mask_.count();
    }
    XDL_CHECK(stat.w_ > 0);
    tstat.k_ = opt->table();
    tstat.seq_.push_back(&stat);
  }

  XDL_CHECK(feature_stats_.size() != 0);

  return true;
}

bool PackFeature::Setup() {
  batch_->ts_count_ = 0;
  for (auto &it: feature_stats_) {
    const std::string &name = it.first;
    auto &stat = it.second;
    auto ktable = stat.opt_->table();

    stat.blk_ = batch_->GetMutable(name);
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
      size_t N = TableSize(stat);
      auto segment = new Tensor(dev_, TensorShape({N}), types::kInt32);
      stat.blk_->ts_[Block::kSegment] = segment;

      stat.blk_->ts_count_ = 3;
      batch_->ts_count_ += stat.blk_->ts_count_;
      //XDL_LOG(DEBUG) << "create " << name << ".val " << value->Shape()
      //    << " " << name << ".idx " << key->Shape()
      //    << " " << name << ".seg " << segment->Shape();
    } else {
      if (stat.blk_->ts_[Block::kValue] != nullptr) {
        delete stat.blk_->ts_[Block::kValue];
      }
      auto value = new Tensor(dev_, TensorShape({TableSize(stat), stat.w_}), types::kFloat);
      stat.blk_->ts_[Block::kValue] = value;
      stat.blk_->ts_count_ = 1;
      batch_->ts_count_ += stat.blk_->ts_count_;
      //XDL_LOG(DEBUG) << "create " << name << ".val " << value->Shape();
    }
  }

  /// indicator
  for (int k = 0; k < table_stats_.size() && TableN(k+1) > 0; ++k) {
    XDL_DCHECK(k < table_stats_.size());
    auto& tstat = table_stats_[k];
    const std::string &name = kIndicatorPrefix + std::to_string(k);
    XDL_DCHECK(tstat.n_ > 0);
    if (tstat.blk_ == nullptr) {
      auto blk = batch_->GetMutable(name);
      tstat.blk_ = blk;
    }

    auto blk = tstat.blk_;
    if (blk->ts_[Block::kIndex] != nullptr) {
      delete blk->ts_[Block::kIndex];
    }
    auto indicator = new Tensor(dev_, TensorShape({TableSize(k)}), types::kInt32);
    blk->ts_[Block::kIndex] = indicator;
    blk->ts_count_ = 1;
    batch_->ts_count_ += blk->ts_count_;
    //XDL_LOG(DEBUG) << "create " << name << " " << indicator->Shape();
  }
  return true;
}

std::pair<int, int> PackFeature::Stat(const PParam &pparam) {
  XDL_CHECK(pparam.ftable_ != nullptr);

  auto ktable = pparam.ktable_;
  auto ftable = pparam.ftable_;
  XDL_CHECK(pparam.end_ <= ftable->feature_lines_size()) << "ktable=" << ktable
      << " pparam.end=" << pparam.end_ << " feature_lines=" << ftable->feature_lines_size();

  int ref_l = INT_MAX, ref_r = INT_MIN;
  int begin = std::max(pparam.begin_, 0);
  int end = std::min(pparam.end_, ftable->feature_lines_size());

  //XDL_LOG(DEBUG) << "stat[" << pparam.isgroup_ << ", " << ktable <<  "] (0)" << pparam.begin_ <<
  //    " -> " <<  pparam.end_ << "(" << ftable->feature_lines_size() << ")";
  XDL_DCHECK(ktable < table_stats_.size());
  auto &tstat = table_stats_[ktable];

  for (int n = begin; n < end; ++n) {
    auto &fl = ftable->feature_lines(n);
    if (ktable < schema_->ntable() - 1) {
      XDL_CHECK(fl.has_refer()) << "ktable=" << ktable << " ntable=" << schema_->ntable();
      ref_l = std::min(ref_l, fl.refer());
      ref_r = std::max(ref_r, fl.refer());
    }

    /// foreach feature
    for (auto &f: fl.features()) {
      auto it = feature_stats_.find(f.name());
      if (it == feature_stats_.end()) {
        continue;
      }

      auto &stat = it->second;
      XDL_CHECK(stat.opt_->table() == ktable) << "feature=" << f.name()
          << " opt=" << stat.opt_
          << " opt.table=" << stat.opt_->table() << " ktable=" << ktable;

      if (stat.opt_->type() == kDense) {
        XDL_CHECK(f.values_size() == 1) << "dense feature=" << stat.opt_->name() 
            << " must be presented as vector of one value, value size=" << f.values_size();
      }
      size_t vcount = f.values_size();
      if (stat.opt_->has_cutoff() && stat.opt_->cutoff() != 0 && abs(stat.opt_->cutoff()) < vcount) {
        stat.n_ += abs(stat.opt_->cutoff());
      } else {
        stat.n_ += f.values_size();
      }
    }
    ++tstat.n_;
  }

  XDL_CHECK(tstat.n_ <= schema_->batch_size_) << "table[" << ktable << "]" << " n="
      << tstat.n_ << " batch_size=" << schema_->batch_size_;

  //XDL_LOG(DEBUG) << "stat[" << pparam.isgroup_ << ", " << ktable <<  "] " << ref_l <<
  //    " -> " <<  ref_r + 1;
  return {ref_l, ref_r+1};
}

size_t PackFeature::OnFeature(FStat *stat, off_t offset, const Feature &f, int cutoff) {
  auto blk = stat->blk_;

  /// foreach feature value
  bool sparse = stat->opt_->type() == kSparse;
  if (!sparse) {
    XDL_CHECK(f.values_size() == 1);
  }

  XDL_CHECK(cutoff != 0);
  size_t vcount = std::min(f.values_size(), abs(cutoff));
  if (vcount < f.values_size()) {
    XDL_LOG(DEBUG) << "cutoff feature=" << stat->opt_->name() << " cutoff=" << cutoff << "/" << f.values_size();
  }
  size_t vbegin = cutoff > 0 ? 0 : std::max(f.values_size()+cutoff, 0);

  for (size_t i = 0; i < vcount; ++i) {
    const auto &v = f.values(i+vbegin);
    if (sparse) {
      XDL_CHECK(v.has_key());
      XDL_CHECK(blk->ts_[Block::kKey] != nullptr && blk->ts_[Block::kSegment] != nullptr);
      auto keys = blk->ts_[Block::kKey]->Raw<int64_t >();
      if (stat->opt_->serialized()) {
        keys[stat->off_] = v.key();
      } else {
        keys[stat->off_*2] = v.hkey();
        keys[stat->off_*2+1] = v.key();
      }
      sparse = true;
    } else {
      XDL_CHECK(stat->off_ <= offset) << "feature=" << stat->opt_->name() <<
          " stat.off=" << stat->off_ << " != table.off=" << offset;
    }

    if (v.has_value()) {
      XDL_CHECK(blk->ts_[Block::kValue] != nullptr);
      auto values = blk->ts_[Block::kValue]->Raw<float>();
      values[stat->off_] = v.value();
    } else if (v.vector_size() > 0) {
      XDL_CHECK(blk->ts_[Block::kValue] != nullptr);
      XDL_DCHECK(stat->opt_->nvec() == v.vector_size()) 
          << "dense feature=" << stat->opt_->name() << " vector_size=" << v.vector_size()
          << " nvec=" << stat->opt_->nvec()  << " width=" << stat->w_;
      auto values = blk->ts_[Block::kValue]->Raw<float>();
      int p = 0;
      for (int m = 0; m < v.vector_size(); ++m) {
        if (stat->opt_->has_mask() && !stat->mask_.test(m)) {
          continue;
        }
        values[offset*stat->w_+p] = v.vector(m);
        ++p;
      }
      XDL_CHECK(p == stat->w_) << "dense feature=" << stat->opt_->name()
            << " vector_size=" << v.vector_size() << " p=" << p << " width=" << stat->w_;
    } else if (!v.has_value()) {
      XDL_CHECK(blk->ts_[Block::kValue] != nullptr);
      auto values = blk->ts_[Block::kValue]->Raw<float>();
      values[stat->off_] = 1.0;
    }
    ++stat->off_;
  }  // feature_value

  if (sparse) {
    auto segments = blk->ts_[Block::kSegment]->Raw<int32_t>();
    segments[offset] = stat->off_;
  }

  return vcount;
}

bool PackFeature::FeaturePad(FStat *stat, size_t offset, size_t end) {
  XDL_DCHECK(offset < end) << "offset=" << offset << " end=" << end;
  if (stat->opt_->type() == kSparse) {
    XDL_CHECK(stat->off_ <= stat->n_) << "feature=" << stat->opt_->name()
        << " off=" << stat->off_ << " n=" << stat->n_ << " end=" << end;

    auto segment = stat->blk_->ts_[Block::kSegment];
    XDL_DCHECK(segment != nullptr && end <= segment->Shape()[0])
        << "shape=(" << segment->Shape()[0] << ") end=" << end;
    auto segments = segment->Raw<int32_t>();
    for (int p = offset; p < end; ++p) {
      segments[p] = stat->off_;
    }
  } else {
    XDL_CHECK(stat->off_ <= offset) << "feature=" << stat->opt_->name()
        << " stat.off=" << stat->off_ << " != offset=" << offset;

    auto value = stat->blk_->ts_[Block::kValue];
    XDL_DCHECK(value != nullptr && end <= value->Shape()[0] && value->Shape()[1] == stat->w_);
    for (int p = offset; p < end; ++p) {
      auto values = value->Raw<float>();
      for (int m = 0; m < stat->w_; ++m) {
        values[p*stat->w_+m] = 0;
      }
    }
  }

  return true;
}

bool PackFeature::OnIndicator(TStat *stat, const FeatureLine &fl) {
  auto indicator = stat->blk_->ts_[Block::kIndex];
  XDL_DCHECK(indicator != nullptr && indicator->Shape()[0]>= stat->n_)
      << "tstat.n=" << stat->n_ << " > indicator."
      << stat->k_ << "(" << indicator->Shape()[0] << ")";
  auto indicators = indicator->Raw<uint32_t>();
  indicators[stat->off_] = stat[1].off_ + fl.refer(); 
  return true;
}

bool PackFeature::IndicatorPad(TStat *stat, off_t end) {
  auto indicator = stat->blk_->ts_[Block::kIndex];
  XDL_DCHECK(indicator != nullptr && end <= indicator->Shape()[0])
      << indicator->Shape()[0] << " != " << end;
  auto indicators = indicator->Raw<uint32_t>();
  int padding_refer = indicators[stat->off_-1]+1;
  for (int i = stat->off_; i < end; ++i) {
    //indicators[i] = padding_refer;
    indicators[i] = indicators[stat->off_-1];  // TODO: should be indicators[i] = padding_refer;
  }
  return true;
}

std::pair<int, int> PackFeature::Run(const PParam &pparam) {
  XDL_CHECK(pparam.ftable_ != nullptr && feature_stats_.size() != 0)
      << "ftable=" << pparam.ftable_ << " stats.size=" << feature_stats_.size();

  auto ktable = pparam.ktable_;
  auto ftable = pparam.ftable_;
  size_t table_size = TableSize(ktable);

  XDL_CHECK(pparam.end_ <= ftable->feature_lines_size());

  int ref_l = INT_MAX, ref_r = INT_MIN;
  int begin = std::max(pparam.begin_, 0);
  int end = std::min(pparam.end_, ftable->feature_lines_size());

  //XDL_LOG(DEBUG) << "run[" << pparam.isgroup_ << ", " << ktable <<  "] (0)" << pparam.begin_ <<
  //    " -> " <<  pparam.end_ << "(" << ftable->feature_lines_size() << ")";
  XDL_DCHECK(ktable < table_stats_.size());
  auto &tstat = table_stats_[ktable];

  for (int n = begin; n < end; ++n) {
    auto &fl = ftable->feature_lines(n);
    if (fl.has_refer()) {
      ref_l = std::min(ref_l, fl.refer());
      ref_r = std::max(ref_r, fl.refer());
    }

    std::vector<int8_t> feature_hits(tstat.seq_.size());
    memset(&feature_hits[0], feature_hits.size(), 0);

    /// foreach feature
    for (auto &f: fl.features()) {
      auto it = feature_stats_.find(f.name());
      if (it == feature_stats_.end()) {
        continue;
      }
      auto &stat = it->second;
      XDL_CHECK(stat.opt_->table() == ktable);

      //XDL_LOG(DEBUG) << "feature["<< stat.seq_ << "]=" << stat.opt_->name() << " stat.off_=" << stat.off_;

      XDL_CHECK(stat.seq_ < feature_hits.size());
      feature_hits[stat.seq_] = 1;

      size_t vcount = OnFeature(&stat, tstat.off_, f, stat.opt_->cutoff()!=0?stat.opt_->cutoff():INT_MAX);
    }  // feature

    // miss
    for (size_t i = 0; i < feature_hits.size(); ++i) {
      if (feature_hits[i] > 0) {
        //auto &stat = *(table_stats_[ktable].seq_[i]);
        //XDL_LOG(DEBUG) << "hit feature["<< stat.seq_ << "]=" << stat.opt_->name() << " stat.off_=" << stat.off_;
        continue;
      }
      XDL_DCHECK(ktable < table_stats_.size());
      auto &stat = *(table_stats_[ktable].seq_[i]);

      //XDL_LOG(DEBUG) << "missed feature["<< stat.seq_ << "]=" << stat.opt_->name() << " stat.off_=" << stat.off_;
      XDL_CHECK(FeaturePad(&stat, tstat.off_, tstat.off_+1));
    }

    if (fl.has_refer()) {
      XDL_CHECK(ktable < kTablesMax - 1 && TableN(ktable+1) > 0) 
          << "tstats[" << ktable << "].n=" << TableN(ktable) << " -> "
          << "tstats[" << ktable+1 << "].n=" << TableN(ktable+1);
      XDL_CHECK(OnIndicator(&tstat, fl));
    }

    ++tstat.off_;
    //XDL_LOG(DEBUG) << "table[" << ktable << "].off=" << table_stats_[ktable].off_;
  } /// for each feature_line


  /// padding main table & indicator
  if (tstat.off_ == tstat.n_) {
    if (/* ktable == 0 && */schema_->padding_ && tstat.n_ < table_size) {
      XDL_LOG(DEBUG) << "batch finish, ktable=" << ktable 
          << " padding " << tstat.n_ << " -> " << table_size;

      for (auto &kv: tstat.seq_) {
        auto &stat = *kv;
        XDL_CHECK(FeaturePad(&stat, tstat.off_, table_size));
      }  /// for each stat in tstat.seq_

      /// padding indicator to the padding line
      if (TableN(ktable + 1) > 0) {
        XDL_CHECK(IndicatorPad(&tstat, table_size));
      }
    }
  }

  return {ref_l, ref_r+1};
}


}  // namespace io
}  // namespace xdl

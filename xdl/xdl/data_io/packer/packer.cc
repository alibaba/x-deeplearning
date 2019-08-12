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

#include "xdl/data_io/packer/packer.h"
#include "xdl/data_io/packer/pack_skey.h"
#include "xdl/data_io/packer/pack_label.h"
#include "xdl/data_io/packer/pack_feature.h"

#include "xdl/data_io/constant.h"
#include "xdl/data_io/pool.h"

#include "xdl/core/lib/timer.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

Packer::Packer(const Schema *schema, Device *dev) 
    : schema_(schema), dev_(dev) {
  packs_[kPackSKey] = new PackSKey(dev_, schema_);
  packs_[kPackLabel] = new PackLabel(dev_, schema_);
  packs_[kPackFeature] = new PackFeature(dev_, schema_);
}

bool Packer::Init() {
  return true;
}

Batch *Packer::Run(std::vector<SGroup*> *sgroups, size_t *total_size_p) {
  //XDL_TIMER_SCOPE(packer_run);
  size_t total_size = *total_size_p;
  if (total_size == 0) {
    return nullptr;
  }
  assert(total_size <= schema_->batch_size_);
  /// init
  Batch *batch = BatchPool::Get()->Acquire();
  if (total_size < schema_->batch_size_) {
    batch->Abandon(true);
  }

  batch->ts_count_ = 0;
  for (int i = 0; i < kPackCount; ++i) {
    packs_[i]->Init(batch);
  }

  /// stat
  PParam pparam;
  for (int i = 0; i < sgroups->size(); ++i) {
    auto sgroup = sgroups->at(i);
    auto sg = sgroup->Get();

    assert(sgroup->end_ > sgroup->begin_);
    total_size -= (sgroup->end_ - sgroup->begin_);

    pparam.ntable_ = sg->feature_tables_size();
    pparam.begin_ = sgroup->begin_;
    pparam.end_ = sgroup->end_;
    pparam.isgroup_ = i;

    pparam.sample_ids_ = &sg->sample_ids();
    packs_[kPackSKey]->Stat(pparam);

    pparam.labels_ = &sg->labels();
    packs_[kPackLabel]->Stat(pparam);

    for (int k = 0; k < schema_->ntable(); ++k) {
      /// padding for fully empty aux feature table & feature line
      if (k == sg->feature_tables_size()) {
        sg->add_feature_tables();
      }

      auto ftable = sg->mutable_feature_tables(k);
      if (ftable->feature_lines_size() == 0) {
        XDL_CHECK (k == sg->feature_tables_size() - 1) << "k=" << k << " table_size=" << sg->feature_tables_size();
        ftable->add_feature_lines();
        XDL_LOG(DEBUG) << "padding empty feature_line for table[" << k << "] ";
        ftable = sg->mutable_feature_tables(k-1);
        for (int n = 0; n < ftable->feature_lines_size(); ++n) {
          auto fl = ftable->mutable_feature_lines(n);
          if (!fl->has_refer()) {
            fl->set_refer(0);
          }
        }
      }

      pparam.ftable_ = &sg->feature_tables(k);
      pparam.ktable_ = k;
      auto range = packs_[kPackFeature]->Stat(pparam);
      pparam.begin_ = range.first;
      pparam.end_ = range.second;
    }
  }
  assert(total_size == 0);

  /// setup
  for (int p = 0; p < kPackCount; ++p) {
    packs_[p]->Setup();
  }

  /// run
  for (int i = 0; i < sgroups->size(); ++i) {
    auto sgroup = sgroups->at(i);
    auto sg = sgroup->Get();

    pparam.ntable_ = sg->feature_tables_size();
    pparam.begin_ = sgroup->begin_;
    pparam.end_ = sgroup->end_;
    pparam.isgroup_ = i;

    pparam.sample_ids_ = &sg->sample_ids();
    packs_[kPackSKey]->Run(pparam);

    pparam.labels_ = &sg->labels();
    packs_[kPackLabel]->Run(pparam);

    XDL_CHECK(sg->feature_tables_size() == schema_->ntable());
    for (int k = 0; k < sg->feature_tables_size(); ++k) {
      pparam.ftable_ = &sg->feature_tables(k);
      pparam.ktable_ = k;
      auto range = packs_[kPackFeature]->Run(pparam);
      pparam.begin_ = range.first;
      pparam.end_ = range.second;
    }

    if (schema_->keep_sgroup_) {
      batch->Keep(sgroup);
    } else {
      SGroupPool::Get()->Release(sgroup);
    }
  }

  sgroups->clear();
  *total_size_p = 0;

  return batch;
}

std::vector<Batch *>Packer::Run(SGroup *sgroup) {
  std::vector<Batch *> out;
  assert(sgroup != nullptr);

  if (sgroup != END) {
    XDL_CHECK(sgroup->end_ > sgroup->begin_);
    XDL_DLOG(DEBUG) << "sgroups.size=[" << sgroups_.size() << "], total_size="  << total_size_
        << ", add sgroup[" << sgroup->begin_ << "," << sgroup->end_ << ")";
    sgroups_.push_back(sgroup);
    total_size_ += sgroup->end_ - sgroup->begin_;
  }

  while (total_size_ >= schema_->batch_size_ ||
         (total_size_ > 0 && sgroup == END)) {
    assert(sgroups_.size() != 0);
    SGroup *tail = nullptr;

    /// fits batch size
    if (total_size_ > schema_->batch_size_) {
      assert(sgroup != END);
      if (schema_->split_group_) {
        /// reduce end
        int oend = sgroup->end_;
        sgroup->end_ -= total_size_ - schema_->batch_size_;
        assert(sgroup->end_ > sgroup->begin_);

        tail = SGroupPool::Get()->Acquire();
        tail->CloneTail(sgroup, oend);
        XDL_DLOG(DEBUG) << "split group, total=" << total_size_ << ", begin="
            << tail->begin_ << ", end=" << tail->end_ << ", size="
            << tail->size_;
        total_size_ = schema_->batch_size_;
      } else if (sgroups_.size() > 1) {
        /// if not only, pop back
        tail = sgroups_.back();
        sgroups_.pop_back();
        XDL_DLOG(DEBUG) << "pop group, total=" << total_size_ << ", begin="
            << sgroup->begin_ << ", end=" << sgroup->end_ << ", size="
            << sgroup->size_;
        total_size_ -= sgroup->end_ - sgroup->begin_;
      } else {
        /// if only, trancate
        assert(sgroups_.size() == 1);
        assert(sgroup->end_ - sgroup->begin_ > schema_->batch_size_);
        sgroup->end_ = sgroup->begin_ + schema_->batch_size_;
        XDL_DLOG(DEBUG) << "trancate group, total=" << total_size_ << ", begin="
            << sgroup->begin_ << ", end=" << sgroup->end_ << ", size="
            << sgroup->size_;
        total_size_ = schema_->batch_size_;
      }
    }

    assert(total_size_ <= schema_->batch_size_);

    /// batching
    Batch *batch = Run(&sgroups_, &total_size_);
    assert(batch != nullptr);
    assert(sgroups_.size() == 0 && total_size_ == 0);
    out.push_back(batch);

    /// leave tail for next
    if (tail != nullptr) {
      sgroups_.push_back(tail);
      total_size_ = tail->end_ - tail->begin_;
      sgroup = tail;
    }
  }
  return out;
}

}  // namespace io
}  // namespace xdl

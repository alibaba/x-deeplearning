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

#include "xdl/data_io/packer/pack_label.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

std::pair<int, int> PackLabel::Stat(const PParam &pparam) {
  XDL_CHECK(pparam.labels_ != nullptr);

  auto labels = pparam.labels_;
  int begin = std::max(pparam.begin_, 0);
  int end = std::min(pparam.end_, labels->size());

  XDL_CHECK(end - begin <= schema_->batch_size_) << "n=" << (end - begin)
      << " schema.batch_size=" << schema_->batch_size_;

  for (int n = begin; n < end; ++n) {
    auto &label = labels->Get(n);
    size_t label_count = label.values_size();
    if (label_count_ == 0) {
      label_count_ = label_count;
    } else {
      XDL_CHECK(label_count == label_count_) << label_count << " " << label_count_;
    }
    ++ n_;
  }
  XDL_CHECK(label_count_ >= schema_->label_count_) << "label.values_size=" << label_count_
      << " schema.label_count=" << schema_->label_count_;
  if (label_count_ > schema_->label_count_) {
    XDL_LOG(WARNING) << "label.values_size=" << label_count_
      << " schema.label_count=" << schema_->label_count_;
  }

  return {0, 0};
}

bool PackLabel::Init(Batch *batch) {
  Pack::Init(batch);
  n_ = 0;
  offset_ = 0;
  return true;
}

bool PackLabel::Setup() {
  XDL_CHECK(label_count_ != 0);
  size_t batch_size = schema_->padding_?schema_->batch_size_:n_;

  auto blk = batch_->GetMutable(kLabelName);
  if (blk->ts_[Block::kValue] != nullptr) {
    delete blk->ts_[Block::kValue];
  }
  blk->ts_[Block::kValue] = new Tensor(dev_, TensorShape({batch_size, label_count_}), types::kFloat);

  blk_ = blk;
  blk_->ts_count_ = 1;
  batch_->ts_count_ += blk_->ts_count_;
  return true;
}

std::pair<int, int> PackLabel::Run(const PParam &pparam) {
  XDL_CHECK(pparam.labels_ != nullptr && label_count_ > 0 && blk_ != nullptr);
  size_t batch_size = schema_->padding_?schema_->batch_size_:n_;

  XDL_CHECK(blk_->ts_[Block::kValue] != nullptr);
  auto value = (blk_->ts_[Block::kValue]->Raw<float>());

  auto &labels = pparam.labels_;
  int begin = std::max(pparam.begin_, 0);
  int end = std::min(pparam.end_, labels->size());

  for (int n = begin; n < end; ++n) {
    auto &label = labels->Get(n);
    XDL_CHECK(label.values_size() == label_count_);
    for (int v = 0; v < label_count_; ++v, ++offset_) {
      XDL_CHECK(offset_ <= n_ * label_count_);
      auto val = label.values(v);
      value[offset_] = val;
    }
  }

  //padding
  if (schema_->padding_ && offset_== n_*label_count_ && n_ < batch_size) {
    for (int n = n_; n < batch_size; ++n) {
      for (int v = 0; v < label_count_; ++v, ++offset_) {
        XDL_CHECK(offset_ <= batch_size * label_count_);
        value[offset_] = 0;
      }
    }
  }
  return {0, 0};
}

}  // namespace io
}  // namespace xdl

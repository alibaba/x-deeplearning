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

#include "xdl/data_io/packer/pack_skey.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

std::pair<int, int> PackSKey::Stat(const PParam &pparam) {
  XDL_CHECK(pparam.sample_ids_ != nullptr);

  auto sample_ids = pparam.sample_ids_;
  int begin = std::max(pparam.begin_, 0);
  int end = std::min(pparam.end_, sample_ids->size());

  for (int n = begin; n < end; ++n) {
    auto &skey = sample_ids->Get(n);
    skey_len_max_ = std::max(skey_len_max_, skey.size() + 1);
    ++ n_;
  }
  XDL_CHECK(n_ <= schema_->batch_size_);

  return {0, 0};
}

bool PackSKey::Init(Batch *batch) {
  Pack::Init(batch);
  skey_len_max_ = 1;
  n_ = 0;
  offset_ = 0;
  return true;
};

bool PackSKey::Setup() {
  size_t batch_size = schema_->padding_?schema_->batch_size_:n_;

  auto blk = batch_->GetMutable(kSKeyName);
  XDL_CHECK(blk != nullptr);

  if (blk->ts_[Block::kSBuf] != nullptr) {
    delete blk->ts_[Block::kSBuf];
  }
  blk->ts_[Block::kSBuf] = new Tensor(dev_, TensorShape({batch_size, skey_len_max_}), types::kInt8);

  blk_ = blk;
  blk_->ts_count_ = 1;
  batch_->ts_count_ += blk_->ts_count_;
  return true;
}

std::pair<int, int> PackSKey::Run(const PParam &pparam) {
  XDL_CHECK(pparam.sample_ids_ != nullptr && skey_len_max_ > 0 && blk_ != nullptr)
      << "skey_len_max=" << skey_len_max_ << ", blk=" << (void *)blk_;
  size_t batch_size = schema_->padding_?schema_->batch_size_:n_;

  auto sbuf = (char *)blk_->ts_[Block::kSBuf]->Raw<int8_t>();

  auto sample_ids = pparam.sample_ids_;
  int begin = std::max(pparam.begin_, 0);
  int end = std::min(pparam.end_, sample_ids->size());

  for (int n = begin; n < end; ++n) {
    XDL_CHECK(offset_ <= n_) << offset_ << " " << n_;
    auto &skey = sample_ids->Get(n);
    XDL_CHECK(skey.size() < skey_len_max_);
    strcpy(&sbuf[skey_len_max_*offset_], skey.c_str());
    ++ offset_;
  }

  //padding
  if (schema_->padding_ && offset_ == n_ && n_ < batch_size) {
    for (; offset_ < batch_size; ++offset_) {
      sbuf[skey_len_max_*offset_] = '\0';
    }
  }

  return {0, 0};
}

}  // namespace io
}  // namespace xdl

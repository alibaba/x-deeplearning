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

#include "xdl/data_io/parser/parse_v4.h"

#include <assert.h>

#include "xdl/data_io/pool.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

bool ParseV4::InitMeta(const std::string &contents) {
  XDL_CHECK(meta_.ParseFromString(contents))
      << "parse sample meta failed, contents=" << contents;

  XDL_LOG(DEBUG) << meta_.DebugString();

  for (int i = 0; i < meta_.data_block_meta_size(); ++i) {
    std::vector<std::string> *fea = nullptr;
    auto &bm = meta_.data_block_meta(i);
    if (bm.data_block_type() == v4::kNCommonFeature) {
      fea = &ncomm_;
    } else if (bm.data_block_type() == v4::kCommonFeature) {
      fea = &comm_;
    } else {
      XDL_LOG(FATAL) << "unknown meta type=" << bm.data_block_type();
    }
    XDL_CHECK(bm.feature_group_meta_size() > 0);
    for (int j = 0; j < bm.feature_group_meta_size(); ++j) {
      auto &fgm = bm.feature_group_meta(j);
      fea->push_back(fgm.feature_group_name());
    }
  }

  XDL_CHECK(ncomm_.size() > 0);
  XDL_CHECK(comm_.size() > 0);
  return true;
}

ssize_t ParseV4::GetSize(const char *str, size_t len) {
  size_t size = ((uint32_t *)str)[0];
  size += sizeof(uint32_t);
  if (size > len) {
    return -1;
  }
  return size;
}

SGroup *ParseV4::Run(const char *str, size_t len) {
  if (str == nullptr || len == 0) {
    return nullptr;
  }

  v4::SampleGroup v4sg;
  XDL_CHECK(v4sg.ParseFromArray(str + sizeof(uint32_t), len - sizeof(uint32_t)))
      << "parse sample group failed, len=" << len;

  SGroup *sgroup = SGroupPool::Get()->Acquire();
  SampleGroup *sg = sgroup->New();

  for (int i = 0; i < v4sg.data_block_size(); ++i) {
    auto &block = v4sg.data_block(i);
    switch (block.data_block_type()) {
      case v4::kSampleInfo:
        OnSKey(block, sg);
        break;
      case v4::kLabel:
        OnLabel(block, sg);
        break;
      case v4::kNCommonFeature:
        if (sg->feature_tables_size() == 0) {
          sg->add_feature_tables();
        }
        OnTable(block, sg->mutable_feature_tables(0));
        break;
      case v4::kCommonFeature:
        while (sg->feature_tables_size() < 2) {
          sg->add_feature_tables();
        }
        OnTable(block, sg->mutable_feature_tables(1));
        break;
      default:
        XDL_LOG(FATAL) << "unknown block type=" << block.data_block_type();
    }
  }

  size_t n = sg->labels_size();
  if (n == 0) {
    SGroupPool::Get()->Release(sgroup);
    return nullptr;
  }

  XDL_CHECK(sg->sample_ids_size() == 0 || sg->sample_ids_size() == n)
      << "sample_id.size=" << sg->sample_ids_size() << " n=" << n;
  XDL_CHECK(sg->feature_tables(0).feature_lines_size() == n)
      << "table[0].size=" << sg->feature_tables(0).feature_lines_size() << " n=" << n;

  sgroup->Reset();
  return sgroup;
}

bool ParseV4::OnLabel(const v4::DataBlock &block, SampleGroup *sg) {
  for (int i = 0; i < block.label_block_size(); ++i) {
    auto &lb = block.label_block(i);
    auto lb_ = sg->add_labels();
    for (int j = 0; j < lb.data_size(); ++j) {
      lb_->add_values(lb.data(j));
    }
  }
  return true;
}

bool ParseV4::OnSKey(const v4::DataBlock &block, SampleGroup *sg) {
  for (int i = 0; i < block.sample_info_block_size(); ++i) {
    auto &sb = block.sample_info_block(i);
    sg->add_sample_ids(sb.info());
  }
  return true;
}

bool ParseV4::OnTable(const v4::DataBlock &block, FeatureTable *tab) {
  std::vector<std::string> *names = nullptr;
  if (block.data_block_type() == v4::kNCommonFeature) {
    names = &ncomm_;
  } else if (block.data_block_type() == v4::kCommonFeature) {
    XDL_CHECK(block.feature_block_size() == 1);
    names = &comm_;
  }

  for (int i = 0; i < block.feature_block_size(); ++i) {
    auto &fb = block.feature_block(i);
    auto fl_ = tab->add_feature_lines();
    for (int j = 0; j < fb.feature_group_size(); ++j) {
      auto &fg = fb.feature_group(j);
      auto fg_ = fl_->add_features();
      fg_->set_type(kSparse);
      unsigned fi = fg.feature_index();
      XDL_CHECK(fi < names->size());
      fg_->set_name(names->at(fi));
      for (int k = 0; k < fg.kv_feature_size(); ++k) {
        auto &kv = fg.kv_feature(k);
        auto kv_ = fg_->add_values();
        kv_->set_key(kv.id());
        kv_->set_value(kv.value());
      }
    }
    if (block.data_block_type() == v4::kNCommonFeature) {
      fl_->set_refer(0);
    }
  }
  return true;
}

}  // namespace xdl
}  // namespace io

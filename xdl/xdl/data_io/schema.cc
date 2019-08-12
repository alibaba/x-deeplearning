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

#include "xdl/data_io/schema.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

const FeatureOption *Schema::Get(const std::string &name) const {
  auto const it = feature_opts_.find(name);
  return it == feature_opts_.end() ? nullptr : it->second;
}

const FeatureOption *Schema::Get(const std::string &name, int table) const {
  XDL_CHECK(table < feature_table_.size());
  auto it = feature_table_[table].find(name);
  return it == feature_table_[table].end() ? nullptr : it->second;
}

bool Schema::Add(const FeatureOption *opt) {
  auto it = feature_opts_.find(opt->name());
  XDL_CHECK(it == feature_opts_.end()) << opt->name() <<  " existed"; 
  feature_opts_.insert(std::make_pair(opt->name(), opt));

  XDL_CHECK(opt->table() >= 0 && opt->table() < kTablesMax);
  if ((size_t)opt->table() >= feature_table_.size())  feature_table_.resize(opt->table()+1);
  auto &opts = feature_table_[opt->table()];

  opts.insert(std::make_pair(opt->name(), opt));

  auto &out_list_ = opt->type() == kSparse ? sparse_list_ : dense_list_;
  out_list_.push_back(opt->name());

  return true;
}

const std::vector<FeatureOptionMap> &Schema::feature_table() const {
  return feature_table_;
}

const std::map<std::string, const FeatureOption*> &Schema::feature_opts() const {
  return feature_opts_;
}

const size_t Schema::ntable() const {
  return feature_table_.size();
}

const std::vector<std::string> &Schema::sparse_list() const {
  return sparse_list_;
}

const std::vector<std::string> &Schema::dense_list() const {
  return dense_list_;
}

}  // namespace io
}  // namespace xdl
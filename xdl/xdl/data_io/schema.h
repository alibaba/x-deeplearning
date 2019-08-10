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

#ifndef XDL_IO_SCHEMA_H_
#define XDL_IO_SCHEMA_H_

#include "xdl/proto/feaconf.pb.h"
#include "xdl/core/lib/singleton.h"

#include <map>

namespace xdl {
namespace io {

using FeatureOptionMap = std::map<std::string, const FeatureOption *>;

/// name->opt
class Schema {
 public:
  virtual ~Schema() {
    for (auto iter : feature_opts_)  delete iter.second;
  }

  const FeatureOption *Get(const std::string &name) const;
  const FeatureOption *Get(const std::string &name, int table) const;

  bool Add(const FeatureOption *option);
  
  const FeatureOptionMap &feature_opts() const;
  const std::vector<FeatureOptionMap> &feature_table() const;
  const std::vector<std::string> &sparse_list() const;
  const std::vector<std::string> &dense_list() const;

  const size_t ntable() const;

  size_t batch_size_ = 1024;
  size_t label_count_ = 1;
  bool keep_sgroup_ = false;
  bool keep_skey_ = false;
  bool padding_ = true;
  bool split_group_ = true;
 protected:
  FeatureOptionMap feature_opts_;
  std::vector<FeatureOptionMap> feature_table_;

  std::vector<std::string> sparse_list_;
  std::vector<std::string> dense_list_;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_SCHEMA_H_
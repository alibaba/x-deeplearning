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


#include "xdl/data_io/op/debug_rebuild_op.h"
#include "xdl/core/utils/logging.h"

#include <iostream>

namespace xdl {
namespace io {

bool DebugRebuildOP::Init(const std::map<std::string, std::string> &params) {
  auto it = params.find("limit");
  if (it != params.end()) {
    limit_ = atoi(it->second.c_str());
    XDL_CHECK(limit_> 0);
    std::cout << "limit: " << limit_;
  }

  it = params.find("repeats");
  if (it != params.end()) {
    repeats_ = atoi(it->second.c_str());
    std::cout << "repeats: " << limit_;
  }
  return true;
}

bool DebugRebuildOP::Run(SampleGroup *sample_group) {
  //auto s = sample_group->ShortDebugString();
  //std::cout << "sg: " << s << std::endl;

  auto ft = sample_group->mutable_feature_tables(0);
  int count = ft->feature_lines_size();

  for (int i = 0; i < count; ++i) {
    auto fl = ft->add_feature_lines();
    auto f = fl->add_features();
    f->set_name("unit_id");
    f->set_type(kSparse);
    auto fv = f->add_values();
    fv->set_key(i);
    fv->set_value(i*0.1);
  }

  int lcount = sample_group->labels_size();
  XDL_CHECK(lcount == count);

  for (int i = 0; i < count; ++i) {
    auto lb = sample_group->add_labels();
    lb->add_values(i*0.1);
  }

  lcount = sample_group->sample_ids_size();
  XDL_CHECK(lcount == count);

  for (int i = 0; i < count; ++i) {
    sample_group->add_sample_ids(std::to_string(i));
  }




  std::unique_lock<std::mutex> lck(mutex_);
  return ++repeats_ <= limit_;
}

XDL_REGISTER_IOP(DebugRebuildOP)

}  // namespace io
}  // namespace xdl

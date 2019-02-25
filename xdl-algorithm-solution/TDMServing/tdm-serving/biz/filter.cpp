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

#include "biz/filter.h"
#include "index/item.h"
#include "proto/search.pb.h"
#include "util/log.h"

namespace tdm_serving {

Filter::Filter() {
}

Filter::~Filter() {
}

bool Filter::Init(const std::string& section,
                  const util::ConfParser& /*conf_parser*/) {
  filter_name_ = section;
  return true;
}

bool Filter::IsFiltered(const FilterInfo* filter_info,
                        const Item& /*item*/) {
  if (filter_info == NULL) {
    return false;
  }

  return false;
}



}  // namespace tdm_serving

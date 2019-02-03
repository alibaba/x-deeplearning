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

#ifndef TDM_SERVING_BIZ_FILTER_H_
#define TDM_SERVING_BIZ_FILTER_H_

#include "common/common_def.h"
#include "util/conf_parser.h"
#include "util/registerer.h"

namespace tdm_serving {

class Item;
class FilterInfo;

class Filter {
 public:
  Filter();
  virtual ~Filter();

  // Initialize the filter by parsed config,
  // section specifies the section name of filter in config
  virtual bool Init(const std::string& section,
                    const util::ConfParser& conf_parser);

  // Filter Item
  virtual bool IsFiltered(const FilterInfo* filter_info,
                          const Item& item);

  // Get filter_name
  const std::string& filter_name() {
    return filter_name_;
  }

private:
  std::string filter_name_;

  DISALLOW_COPY_AND_ASSIGN(Filter);
};

// define register
REGISTER_REGISTERER(Filter);
#define REGISTER_FILTER(title, name) \
    REGISTER_CLASS(Filter, title, name)

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_FILTER_H_

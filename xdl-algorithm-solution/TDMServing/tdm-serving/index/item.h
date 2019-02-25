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

#ifndef TDM_SERVING_INDEX_ITEM_H_
#define TDM_SERVING_INDEX_ITEM_H_

#include <inttypes.h>

namespace tdm_serving {

// Item interface,
// user for item filtering and sorting
class Item {
 public:
  Item() {}
  virtual ~Item() {}

  // item id
  virtual uint64_t item_id() const = 0;

  // item score calc by model layer
  virtual float score() const = 0;

  virtual bool has_category() const {
    return false;
  }

  // item category
  virtual int32_t category() const {
    return 0;
  }
};

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_ITEM_H_

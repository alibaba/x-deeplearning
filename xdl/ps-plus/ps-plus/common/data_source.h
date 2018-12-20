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

/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef PS_PLUS_COMMON_DATA_SOURCE_H_
#define PS_PLUS_COMMON_DATA_SOURCE_H_

#include <string>
#include <vector>
#include <memory>

#include "ps-plus/common/status.h"

namespace ps {

struct DataClosure {
  /// The cpu-side temporary space
  char* data;
  size_t length;
};

class DataSource {
 public:
  /// destructor
  virtual ~DataSource() {}

  /// Init data source
  virtual Status Init(int rank, int worker_num, size_t default_value_length) = 0;

  /// get the value of key
  virtual Status Get(int64_t id, DataClosure* closure) = 0;

  /// get the values of some keys
  virtual Status BatchGet(const std::vector<int64_t>& ids, 
                          std::vector<DataClosure>* closures) = 0;

  /// return the default value length
  size_t default_value_length() const { return default_value_length_; }

 protected:
  /// The default value length
  size_t default_value_length_;
};

}  // namespace ps 

#endif  // PS_PLUS_COMMON_DATA_SOURCE_H_

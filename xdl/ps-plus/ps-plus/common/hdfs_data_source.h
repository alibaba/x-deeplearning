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
#ifndef PS_PLUS_COMMON_HDFS_DATA_SOURCE_H_
#define PS_PLUS_COMMON_HDFS_DATA_SOURCE_H_

#include "ps-plus/common/data_source.h"

#include <mutex>
#include <unordered_map>

namespace ps {

class HdfsDataSource : public DataSource {
 public:
  /// constructor
  HdfsDataSource(const std::string& filepath,
                 size_t file_num)
    : filepath_(filepath),
      file_num_(file_num) {}

  /// Init hdfs data source
  Status Init(int rank, int worker_num, size_t default_value_length) override;

  /// get the value of key
  Status Get(int64_t id, DataClosure* closure) override;

  /// get batch values of keys
  Status BatchGet(const std::vector<int64_t>& ids,
                  std::vector<DataClosure>* closures);

  /// get batch values of keys
  void BatchGetV2(const std::vector<int64_t>& ids,
                  std::vector<DataClosure>* closures,
                  std::vector<int64_t>* rst_ids);
 protected:
  Status InitFromFile(const std::string& stream);

  Status status_;
  /// data source file directory
  std::string filepath_;
  /// total data source file number
  uint32_t file_num_;
  /// kv map
  std::unordered_map<int64_t, char*> data_;
  std::vector<std::unique_ptr<char[]>> buffer_;
  /// mutex for init
  std::mutex mu_;
  static constexpr int kBufferCount = 1024;
};

}  // namespace ps

#endif  // PS_PLUS_COMMON_HDFS_DATA_SOURCE_H_

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

#ifndef PS_SERVER_UDF_MANAGER_H_
#define PS_SERVER_UDF_MANAGER_H_

#include "ps-plus/message/udf_chain_register.h"
#include "ps-plus/server/udf.h"
#include "ps-plus/common/qrw_lock.h"

namespace ps {
namespace server {

class UdfChain {
 public:
  ~UdfChain();
  Status BuildFromDef(const UdfChainRegister& def);
  Status Process(UdfContext* ctx);
 private:
  std::vector<Udf*> udfs_;
  std::vector<size_t> output_ids_;
  size_t input_size_;
};

class UdfChainManager {
 public:
  UdfChain* GetUdfChain(size_t hash);
  Status RegisterUdfChain(const UdfChainRegister& def);
 private:
  QRWLock rd_lock_;
  std::unordered_map<size_t, std::unique_ptr<UdfChain>> chain_map_;
};

}
}

#endif


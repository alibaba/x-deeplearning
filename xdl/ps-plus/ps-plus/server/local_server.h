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

#ifndef PS_PLUS_SERVER_LOCAL_SERVER_H_
#define PS_PLUS_SERVER_LOCAL_SERVER_H_

#include "ps-plus/common/status.h"
#include "ps-plus/server/storage_manager.h"
#include "ps-plus/server/udf_manager.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/server/streaming_model_args.h"

#include <mutex>

namespace ps {
namespace server {

// LocalServer is only used in inference

class LocalServer {
 public:
  LocalServer(const std::string& ckpt_path);
  ~LocalServer() = default;
  Status Init();
  Status RegisterUdfChain(const UdfChainRegister& def);
  Status Process(size_t udf, 
                 const std::string& variable_name,
                 const std::vector<Data*>& inputs,
                 std::vector<Data*>* outputs);
  Status Restore(const std::string& ckpt_version);
  Status Save(const std::string& ckpt_version);
  Status GetVariableInfo(const std::string& var_name, VariableInfo* info);
  Status RegisterVariable(const std::string& name, const VariableInfo& info);

 private:
  Status RunUdfChain(size_t udf, 
                     const std::string& variable_name, 
                     const std::vector<Data*>& inputs, 
                     UdfContext* ctx);

  Status LoadCheckPointMeta(const std::string& checkpoint,
                            VariableInfoCollection* info);

 private:
  std::unique_ptr<UdfChainManager> udf_chain_manager_;
  std::unique_ptr<StorageManager> storage_manager_;
  std::mutex var_info_mutex_;
  std::unordered_map<std::string, VariableInfo> var_infos_;
  std::string ckpt_path_;
  StreamingModelArgs streaming_model_args_;
};

}
}

#endif


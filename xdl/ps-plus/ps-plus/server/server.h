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

#ifndef PS_PLUS_SERVER_SERVER_H_
#define PS_PLUS_SERVER_SERVER_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/qrw_lock.h"
#include "ps-plus/server/storage_manager.h"
#include "ps-plus/server/udf_manager.h"
#include "ps-plus/message/version.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/server/streaming_model_utils.h"
#include "ps-plus/server/streaming_model_args.h"
#include "ps-plus/message/streaming_model_infos.h"
#include "ps-plus/message/streaming_model_manager.h"

namespace ps {
namespace server {

class Server {
 public:
  Server(size_t id, const StreamingModelArgs& streaming_model_args);
  Status Init();
  Status RegisterUdfChain(Version ver, const UdfChainRegister& def);
  Status RunUdfChain(Version ver, size_t udf, const std::string& variable_name, const std::vector<Data*>& inputs, UdfContext* ctx);
  Status Save(Version ver, const std::string& checkpoint, const VariableInfoCollection& info);
  Status Restore(Version ver, const VariableInfoCollection& from, const VariableInfoCollection& to);
  Status StreamingDenseVarName(Version ver, DenseVarNames* result);
  Status GatherStreamingDenseVar(Version ver, const DenseVarNames& name, DenseVarValues* result);
  Status TriggerStreamingSparse(Version ver, const int& server_id, const std::string& stream_version);
  Status TriggerStreamingHash(Version ver, const int& server_id, const std::string& stream_version);
 private:
  // Writelocked when restore.
  QRWLock server_lock_;
  std::unique_ptr<UdfChainManager> udf_chain_manager_;
  std::unique_ptr<StorageManager> storage_manager_;
  Version ver_;
  size_t id_;
  StreamingModelArgs streaming_model_args_;
};

}
}

#endif


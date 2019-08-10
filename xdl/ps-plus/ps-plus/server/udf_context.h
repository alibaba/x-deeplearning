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

#ifndef PS_SERVER_UDF_CONTEXT_H_
#define PS_SERVER_UDF_CONTEXT_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/data.h"
#include "ps-plus/common/plugin.h"
#include "ps-plus/common/qrw_lock.h"

#include "ps-plus/server/storage_manager.h"
#include "ps-plus/server/variable.h"
#include "ps-plus/server/streaming_model_args.h"

#include <memory>
#include <vector>


namespace ps {
namespace server {

class UdfContext {
 public:
  UdfContext();
  ~UdfContext();
  size_t DataSize();
  Status ProcessOutputs(const std::vector<size_t>& output_ids);
  const std::vector<Data*>& Outputs();
  Status SetStorageManager(StorageManager* storage_manager);
  Status SetVariable(Variable* variable);
  Status SetVariableName(const std::string& variable_name);
  Status SetLocker(QRWLocker* locker);
  Status SetServerLocker(QRWLocker* locker);
  Status SetStreamingModelArgs(StreamingModelArgs* streaming_model_args);

  Status SetData(size_t id, Data* data, bool need_free);
  Status GetData(size_t id, Data** data);
  Status AddDependency(Data* dependency);
  void RemoveOutputDependency();
  StorageManager* GetStorageManager();
  Variable* GetVariable();
  const std::string& GetVariableName();
  QRWLocker* GetLocker();
  QRWLocker* GetServerLocker();
  StreamingModelArgs* GetStreamingModelArgs();
 private:
  std::vector<Data*> datas_;
  std::vector<Data*> outputs_;
  std::vector<Data*> dependencies_;
  Variable* variable_;
  std::string variable_name_;
  StorageManager* storage_manager_;
  QRWLocker* locker_;
  QRWLocker* server_locker_;
  StreamingModelArgs* streaming_model_args_;
};

}
}

#endif


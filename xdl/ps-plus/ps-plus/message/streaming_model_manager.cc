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

#include "ps-plus/message/streaming_model_manager.h"

namespace ps {

Status StreamingModelManager::GetManager(const std::string& path, StreamingModelManager** manager) {
  std::string protocol;
  size_t pos = path.find("://");
  if (pos == std::string::npos) {
    return Status::ArgumentError("StreamingModelManager: protocol parse error");
  } else {
    protocol = path.substr(0, pos);
  }
  *manager = GetPlugin<StreamingModelManager>(protocol);
  if (*manager == nullptr) {
    return Status::NotFound("StreamingModelManager [" + protocol + "] Not found");
  }
  return Status::Ok();
}

Status StreamingModelManager::OpenWriterAny(const std::string& path, std::unique_ptr<StreamingModelWriter>* writer) {
  StreamingModelManager* manager;
  PS_CHECK_STATUS(GetManager(path, &manager));
  return manager->OpenWriter(path, writer);
}

}


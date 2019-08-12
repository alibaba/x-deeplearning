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

#include "worker_state.h"

namespace ps {

WorkerState::WorkerState() 
  : begin_(0)
  , end_(0)
  , epoch_(0) {
}

WorkerState::WorkerState(
      size_t begin, 
      size_t end, 
      size_t epoch, 
      const std::string& path)
  : begin_(begin)
  , end_(end)
  , epoch_(epoch)
  , path_(path) {
}

}

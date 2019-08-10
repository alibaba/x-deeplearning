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

#ifndef XDL_IO_GLOBAL_SCHEDULER_H_
#define XDL_IO_GLOBAL_SCHEDULER_H_

#include <set>
#include <vector>
#include <list>
#include <string>
#include <atomic>

#include "xdl/core/lib/blocking_queue.h"
#include "xdl/data_io/constant.h"
#include "xdl/data_io/scheduler.h"
#include "xdl/data_io/parser/parser.h"
#include "xdl/proto/io_state.pb.h"

namespace ps {
namespace client {
class BaseClient;
}
}

namespace xdl {
namespace io {

class GlobalScheduler: public Scheduler {
 public:
  GlobalScheduler() = delete;
  GlobalScheduler(FileSystem *fs, 
                  const std::string& name,
                  size_t epochs = 1, 
                  size_t worker_id = 0);
  GlobalScheduler(const std::string& name,
                  FSType fs_type = kLocal, 
                  const std::string& namenode = "", 
                  size_t epochs = 1,
                  size_t worker_id = 0);

  bool SetEpochIsolate(bool value);

  bool Schedule() override;
  ReadParam *Acquire() override;
  void Release(ReadParam *) override;
  bool Store(DSState *ds_state) override;
  bool Restore(const DSState &ds_state) override;

 protected:
  std::string name_;
  std::unordered_set<ReadParam*> using_;
  ps::client::BaseClient* client_;
  bool epoch_isolate_;
  size_t worker_id_;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_GLOBAL_SCHEDULER_H_
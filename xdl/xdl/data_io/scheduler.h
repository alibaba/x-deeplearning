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

#ifndef XDL_IO_SCHEDULER_H_
#define XDL_IO_SCHEDULER_H_

#include <set>
#include <vector>
#include <list>
#include <string>
#include <atomic>

#include "xdl/core/lib/blocking_queue.h"
#include "xdl/data_io/constant.h"
#include "xdl/data_io/parser/parser.h"
#include "xdl/proto/io_state.pb.h"

namespace xdl {
namespace io {

static const size_t kSchedCap = 1024*128;

class Scheduler {
 public:
  Scheduler() = delete;
  Scheduler(FileSystem *fs, size_t epochs=1);
  Scheduler(FSType fs_type=kLocal, const std::string &namenode="", size_t epochs=1);

  virtual bool AddPath(const std::string &path);
  virtual bool SetEpochs(size_t epochs);
  virtual bool SetZType(ZType ztype);
  virtual bool SetShuffle(bool shuffle);

  virtual bool Store(DSState *ds_state);
  virtual bool Restore(const DSState &ds_state);

  virtual bool Schedule();
  virtual ReadParam *Acquire();
  virtual void Release(ReadParam *);
  virtual bool finished() const;
  virtual void Clear();
 protected:
  bool restored_ = false;
  bool shuffle_ = false;
  size_t epochs_ = 1;
  ZType ztype_ = kRaw;
  FileSystem *fs_ = nullptr;
  std::vector<std::string> paths_;

  std::atomic<bool> finished_;
  std::mutex mutex_;
  std::set<ReadParam *> using_;
  BlockingQueue<ReadParam *> rparams_;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_SCHEDULER_H_
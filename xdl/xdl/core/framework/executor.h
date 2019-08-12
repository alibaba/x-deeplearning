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

#ifndef XDL_CORE_FRAMEWORK_EXECUTOR_H_
#define XDL_CORE_FRAMEWORK_EXECUTOR_H_

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/thread_pool.h"
#include "xdl/core/framework/graph_def.h"
#include "xdl/core/framework/simple_executor.h"
#include "xdl/core/framework/run_option.h"

namespace xdl {

class Executor {
 public:
  using Callback = SimpleExecutor::Callback;
  explicit Executor(ThreadPool* thread_pool)
    : thread_pool_(thread_pool) {}
  void Run(const GraphDef& graph, const OutputSpec& output, 
           const RunOption& run_option, Callback done);
 private:
  ThreadPool* thread_pool_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_EXECUTOR_H_


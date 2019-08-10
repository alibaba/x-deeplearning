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

#ifndef XDL_CORE_FRAMEWORK_SIMPLE_EXECUTOR_H_
#define XDL_CORE_FRAMEWORK_SIMPLE_EXECUTOR_H_

#include <atomic>
#include <vector>
#include <functional>

#include "xdl/core/framework/graph.h"
#include "xdl/core/framework/device.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/proto/perf_stats.pb.h"
#include "xdl/core/lib/any.h"
#include "xdl/core/framework/run_option.h"

namespace xdl {

class SimpleExecutor {
 public:
  using ExtraInfo = std::unordered_map<std::string, Any>;
  using Callback = std::function<void(Status, const std::vector<Tensor>&, const ExtraInfo&)>;
  using DoneHandler = std::function<void(Status)>;

  static void Run(Graph* graph, const RunOption& run_option, Callback done, ThreadPool* thread_pool);
  static void Run(Graph* graph, const RunOption& run_option, Callback done);

  void AddDoneHandler(DoneHandler handler_) {
    std::unique_lock<std::mutex> lock(done_handler_mu_);
    done_handler_.push_back(handler_);
  }

  const RunOption& GetRunOption() {
    return run_option_;
  }

 private:
  explicit SimpleExecutor(Graph* graph, const RunOption& run_option, 
                          Callback done, ThreadPool* thread_pool)
    : graph_(graph), run_option_(run_option), 
      done_(done), thread_pool_(thread_pool) {
    if (run_option_.perf) {
      while (perf_stats_.node_stats_size() < graph_->nodes.size() + 1) {
        perf_stats_.add_node_stats();        
      }
    }
  }

  void Run();
  void Init();
  Status InitImpl();
  void Launch(int node_id);
  void LaunchDone(int node_id, OpKernelContext* ctx, Status st);
  void RunDone(int node_id, OpKernelContext* ctx, Status st);
  Status CheckInputs(int node_id, const std::vector<Tensor>& inputs);
  Status CheckOutputs(int node_id, const std::vector<Tensor>& outputs);
  void CheckStatus(Status st);
  void Fail(Status st);
  void UnRef(int node_id);
  void DecreaseRunningCounter();
  void Done();

  inline bool IsPerfOn() {
    return run_option_.perf;
  }

  void PerfSetNodeStart(int node_id);
  void PerfSetNodeEnd(int node_id);
  void PerfSetNameAndOp(int node_id);
  void PerfSetThreadId(int node_id);

  Graph* graph_;
  RunOption run_option_;
  Callback done_;
  ThreadPool* thread_pool_;

  std::vector<std::vector<Tensor>> input_;
  std::unique_ptr<std::atomic<int>[]> ref_;
  std::atomic<int> running_counter_;
  Status status_;
  std::atomic<bool> failed_;

  std::mutex done_handler_mu_;
  std::vector<DoneHandler> done_handler_;
  proto::PerfStats perf_stats_;
};

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_SIMPLE_EXECUTOR_H_


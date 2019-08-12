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

#include "xdl/core/framework/simple_executor.h"

#include <thread>
#include <google/protobuf/text_format.h>

#include "xdl/core/framework/device.h"
#include "xdl/core/utils/time_utils.h"
#include "xdl/core/utils/logging.h"

namespace xdl {

namespace {

class RefClosure : public RefCounted {
 public:
  RefClosure(const std::function<void()> done) : done_(done) {}
  ~RefClosure() { done_(); }
 private:
  std::function<void()> done_;
};

}  // namespace

void SimpleExecutor::Run(Graph* graph, 
                         const RunOption& run_option, 
                         Callback done, 
                         ThreadPool* thread_pool) {
  SimpleExecutor* executor = new SimpleExecutor(
      graph, run_option, done, thread_pool);
  executor->Run();
}

void SimpleExecutor::Run(Graph* graph, 
                         const RunOption& run_option, 
                         Callback done) {
  Run(graph, run_option, done, ThreadPool::Global());
}

void SimpleExecutor::Run() {
  Init();
  if (!status_.IsOk()) {
    Done();
    return;
  }
  running_counter_ = 1;
  for (auto item : graph_->nodes[Graph::kSource].outputs) {
    Launch(item.node_id);
  }
  DecreaseRunningCounter();
}

void SimpleExecutor::Init() {
  status_ = InitImpl();
  failed_ = !status_.IsOk();
}

Status SimpleExecutor::InitImpl() {
  running_counter_ = 0;
  ref_.reset(new std::atomic<int>[graph_->nodes.size()]);
  for (size_t i = 0; i < graph_->nodes.size(); i++) {
    input_.emplace_back(graph_->nodes[i].input_size);
    ref_[i] = graph_->nodes[i].inputs.size();
  }
  return Status::Ok();
}

void SimpleExecutor::Launch(int node_id) {
  if (node_id == Graph::kSink) {
    return;
  }
  if (!failed_) {
    CheckStatus(CheckInputs(node_id, input_[node_id]));
  }
  if (failed_) {
    return;
  }
  running_counter_ += 2;
  OpKernelContext* ctx = new OpKernelContext(&(graph_->nodes[node_id].arg),
                                             this,
                                             std::move(input_[node_id]));
  ctx->SetLaunchDone(
      [this, node_id, ctx](Status st){LaunchDone(node_id, ctx, st);});
  ctx->SetRunDone(
      [this, node_id, ctx](Status st){RunDone(node_id, ctx, st);});

  if (IsPerfOn()) {
    PerfSetNodeStart(node_id);
    PerfSetNameAndOp(node_id);
  }

  graph_->nodes[node_id].arg.device->ScheduleToRun(
      thread_pool_, graph_->nodes[node_id].op.get(), ctx);
}

void SimpleExecutor::LaunchDone(int node_id, OpKernelContext* ctx, Status st) {
  CheckStatus(st);
  auto& outputs = ctx->GetOutputs();
  if (!failed_) {
    CheckStatus(CheckOutputs(node_id, outputs));
  }
  if (!failed_) {
    for (auto&& item : graph_->nodes[node_id].outputs) {
      if (item.input_id == Node::kDependency) {
        UnRef(item.node_id);
        continue;
      }
      // Process on RunDone
      Device* src_device = graph_->nodes[node_id].arg.device;
      Device* dst_device = graph_->nodes[item.node_id].arg.device;
      const std::vector<Device*>& input_devices = graph_->nodes[item.node_id].arg.input_devices;
      if (item.input_id < input_devices.size() && input_devices[item.input_id] != nullptr) {
        dst_device = input_devices[item.input_id];
      }
      if (src_device != dst_device) {
        continue;
      }
      input_[item.node_id][item.input_id] = outputs[item.output_id];
      UnRef(item.node_id);
    }
  }
  DecreaseRunningCounter();
  ctx->UnRef();
}

void SimpleExecutor::RunDone(int node_id, OpKernelContext* ctx, Status st) {
  CheckStatus(st);
  auto& outputs = ctx->GetOutputs();
  if (!failed_) {
    CheckStatus(CheckOutputs(node_id, outputs));
  }
  RefClosure* closure = new RefClosure([this, ctx]{
    DecreaseRunningCounter();
    ctx->UnRef();
  });
  if (!failed_) {
    for (auto&& item : graph_->nodes[node_id].outputs) {
      // Process on LaunchDone
      Device* src_device = graph_->nodes[node_id].arg.device;
      Device* dst_device = graph_->nodes[item.node_id].arg.device;
      const std::vector<Device*>& input_devices = graph_->nodes[item.node_id].arg.input_devices;
      if (item.input_id < input_devices.size() && input_devices[item.input_id] != nullptr) {
        dst_device = input_devices[item.input_id];
      }
      if (src_device == dst_device) {
        continue;
      }
      if (item.input_id != Node::kDependency) {
        auto iter = graph_->device_converter.find(
            std::pair<Device*, Device*>(
            graph_->nodes[node_id].arg.device,
            graph_->nodes[item.node_id].arg.device));
        if (iter == graph_->device_converter.end()) {
          Fail(Status::Internal("Internal Error, Device Converter Error"));
        } else {
          closure->Ref();
          int node_id = item.node_id;
          iter->second->Convert(
              src_device, dst_device,
              outputs[item.output_id], &input_[item.node_id][item.input_id],
              ThreadPool::Global(),
              [closure, this, node_id] (Status st) {
                CheckStatus(st);
                UnRef(node_id);
                closure->UnRef();
              });
        }
      }
    }
  }
  
  if (IsPerfOn()) {
    PerfSetThreadId(node_id);
    PerfSetNodeEnd(node_id);
  }

  closure->UnRef();
}

Status SimpleExecutor::CheckInputs(int node_id,
                                   const std::vector<Tensor>& inputs) {
  auto& types = graph_->nodes[node_id].arg.input_type;
  XDL_CHECK_COND(inputs.size() == types.size(),
                 Status::Internal("Op Run Input Size Mismatch "
                                  + graph_->nodes[node_id].name));
  for (size_t i = 0; i < types.size(); i++) {
    XDL_CHECK_COND(inputs[i].Initialized(),
                   Status::Internal("Op Run Input Tensor Not Initialized "
                                    + graph_->nodes[node_id].name + "id="
                                    + std::to_string(i)));
    XDL_CHECK_COND(inputs[i].Type() == types[i],
                   Status::Internal("Op Run Input Tensor Type Mismatch "
                                    + graph_->nodes[node_id].name + "id="
                                    + std::to_string(i)));
  }
  return Status::Ok();
}

Status SimpleExecutor::CheckOutputs(int node_id,
                                    const std::vector<Tensor>& outputs) {
  auto& types = graph_->nodes[node_id].arg.output_type;
  XDL_CHECK_COND(outputs.size() == types.size(),
                 Status::Internal("Op Run Output Size Mismatch "
                                  + graph_->nodes[node_id].name));
  for (size_t i = 0; i < types.size(); i++) {
    XDL_CHECK_COND(outputs[i].Initialized(),
                   Status::Internal("Op Run Output Tensor Not Initialized "
                                    + graph_->nodes[node_id].name + "id="
                                    + std::to_string(i)));
    XDL_CHECK_COND(outputs[i].Type() == types[i],
                   Status::Internal("Op Run Output Tensor Type Mismatch "
                                    + graph_->nodes[node_id].name + "id="
                                    + std::to_string(i)));
  }
  return Status::Ok();
}

void SimpleExecutor::CheckStatus(Status st) {
  if (!st.IsOk()) {
    Fail(st);
  }
}

void SimpleExecutor::Fail(Status st) {
  if (!failed_.exchange(true)) {
    status_ = st;
  }
}

void SimpleExecutor::UnRef(int node_id) {
  if (--ref_[node_id] == 0) {
    Launch(node_id);
  }
}

void SimpleExecutor::DecreaseRunningCounter() {
  if (--running_counter_ == 0) {
    Done();
  }
}

void SimpleExecutor::Done() {
  for (auto&& item : done_handler_) {
    item(status_);
  }
  if (status_.IsOk()) {
    ExtraInfo info = ExtraInfo();
    std::string perf_info;
    google::protobuf::TextFormat::PrintToString(perf_stats_, &perf_info);
    if (IsPerfOn()) {
      info["PERF_RESULT"] = perf_info;
    }
    done_(status_, input_[Graph::kSink], info);
  } else {
    done_(status_, std::vector<Tensor>(), ExtraInfo());
  }
  delete this;
}

void SimpleExecutor::PerfSetNodeStart(int node_id) {
  perf_stats_.mutable_node_stats(node_id)->set_start_micros(
      TimeUtils::NowMicros());
}

void SimpleExecutor::PerfSetNodeEnd(int node_id) {
  perf_stats_.mutable_node_stats(node_id)->set_end_micros(
      TimeUtils::NowMicros());
}

void SimpleExecutor::PerfSetNameAndOp(int node_id) {
  proto::NodeExecStat* stats = perf_stats_.mutable_node_stats(node_id);
  stats->set_node_name(graph_->nodes[node_id].name);
  stats->set_op(graph_->nodes[node_id].op_name);
}

void SimpleExecutor::PerfSetThreadId(int node_id) {
  auto hasher = std::hash<std::thread::id>();
  perf_stats_.mutable_node_stats(node_id)->set_thread_id(
      hasher(std::this_thread::get_id()));
}

}  // namespace xdl


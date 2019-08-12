/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#include "xdl/core/backend/tf/tf_runner.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <dlfcn.h>
#include <stdlib.h>

namespace xdl {

const std::string TFRunner::TF_RUNNER_ROOT = "TF_RUNNER_ROOT";

TFRunner::TFRunner() : session_(nullptr) {
}

TFRunner::~TFRunner() {
  if (session_ != nullptr) {
    // fixme: we use cuda driver-api and runtime-api together, will cause 
    // tensorflow destroy incorrectly under gpu environment
    // delete session_;
    // session_ = nullptr;
  }
}

Status TFRunner::Init(const std::string& graph_def_pb, float gpu_memory_fraction) {
  tensorflow::ConfigProto config;
  config.set_allow_soft_placement(true);
  config.set_intra_op_parallelism_threads(0);
  config.set_inter_op_parallelism_threads(0);
  config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(gpu_memory_fraction);
  tensorflow::SessionOptions session_options;
  session_options.config = config;
  session_ = NewSession(session_options);
  XDL_CHECK_COND(session_ != nullptr, 
                 Status::Internal("tf new session failed"));
  tensorflow::Status status;
  tensorflow::MetaGraphDef graph_def;
  if (!graph_def.ParseFromString(graph_def_pb)) {
    return Status::Internal("Load graph failed");
  }

  status = session_->Create(graph_def.graph_def());
  if (!status.ok()) {
    return Status::Internal("tf create session failed, errmsg:" + 
                            status.ToString());
  }
  
  return Status::Ok();
}

Status TFRunner::Run(const InputList &inputs,
                     const std::vector<std::string> &ops_names,
                     std::vector<tensorflow::Tensor>* outputs) {
  tensorflow::Status status;
  status = session_->Run(inputs, ops_names, {}, outputs);
  if (!status.ok()) {
    return Status::Internal("tf session run failed, errormsg:" + 
                            status.error_message());
  }
  
  return Status::Ok();
}

} // namespace xdl

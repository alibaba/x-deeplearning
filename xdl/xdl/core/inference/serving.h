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

#ifndef XDL_CORE_INFERENCE_SERVING_H_
#define XDL_CORE_INFERENCE_SERVING_H_

#include "xdl/core/utils/logging.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/framework/cpu_device.h"
#include "xdl/core/framework/executor.h"
#include "xdl/core/ops/ps_ops/client.h"

namespace xdl {

class Serving {
 public:
  Serving(const std::string& ckpt_dir);
  ~Serving() = default;

  Status Init(const std::string& graph_path,
              const std::string& ckpt_version);

  // used when GraphDef has inference TagDef defined
  // in training 
  Status Predict(const Executor::Feeds& feeds, 
                 std::vector<Tensor>* outputs);

  // used in common case
  Status Predict(const Executor::Feeds& feeds, 
                 const std::vector<std::string>& output_op_names,
                 std::vector<Tensor>* results);

 private:
  Status LoadGraph(const std::string& graph_path, 
                   bool text_format = true);
  Status ParseInferenceTag();
  Status Predict(const Executor::Feeds& feeds,
                 const OutputSpec output_spec,
                 std::vector<Tensor>* results);

 private:
  GraphDef graph_;
  std::string ckpt_dir_;
  ps::client::BaseClient* client_;
  std::unique_ptr<Executor> executor_;
  std::unordered_map<std::string, std::string> input_tag_;
  OutputSpec output_spec_;
};

}  // namespace xdl

#endif  // XDL_CORE_INFERENCE_SERVING_H_


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

#ifndef XDL_BACKEND_MXNET_MXNET_RUNNER_H_
#define XDL_BACKEND_MXNET_MXNET_RUNNER_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <mxnet-cpp/MxNetCpp.h>

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/tensor.h"
#include "xdl/core/backend/mxnet/convert_utils.h"

namespace xdl {

class MxnetRunner {
 public:
  using InputList = std::vector<std::pair<std::string, Tensor>>;
  using DataList = std::vector<mxnet::cpp::NDArray>;

  MxnetRunner(bool is_training = true);
  ~MxnetRunner();

  Status Init(const std::string &graph_def, 
              const std::string& device_type);

  // According to MXNet documents,
  // the returned executor shares state with the current one,
  // and cannot be used in parallel with it.
  Status Run(const InputList& inputs, 
             DataList* outputs, 
             DataList* gradients);

  #ifdef USE_GPU
  Status Run(const InputList& inputs, 
             DataList* outputs, 
             DataList* gradients,
             cudaStream_t stream);
  #endif

  void SetInitGrad(std::vector<mxnet::cpp::NDArray>* init_grad) {
    has_init_grad_ = true;
    init_grad_ = init_grad;
  }

 protected:
  Status Bind(mxnet::cpp::Executor *&exec,
              const InputList& inputs);
  bool Reshape(mxnet::cpp::Executor *&exec,
               const InputList& inputs);

 private:
  mxnet::cpp::Symbol loss_;
  mxnet::cpp::Executor* exec_ = nullptr;
  std::string device_type_;
  bool is_training_;
  std::vector<mxnet::cpp::NDArray>* init_grad_;
  std::set<std::string> arg_sets_;
  std::set<std::string> aux_sets_;
  bool has_init_grad_;

  bool enable_reshape_ = true;
  size_t *arg_name_idx_ = nullptr;
  size_t unused_idx_ = 0xFFFFFFFFu;
};

} // namespace xdl

#endif // XDL_BACKEND_MXNET_MXNET_RUNNER_H_



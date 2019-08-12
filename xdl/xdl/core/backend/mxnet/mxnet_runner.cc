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

#include "xdl/core/backend/mxnet/mxnet_runner.h"

#include "xdl/core/utils/logging.h"

namespace xdl {

MxnetRunner::MxnetRunner(bool is_training) 
  : is_training_(is_training) 
  , has_init_grad_(false)
  , init_grad_(nullptr) {
}

MxnetRunner::~MxnetRunner() {
  if (exec_ != nullptr) {
    // There is a bug about Engine::Get() of mxnet, so we don't delete exec_ here until mxnet fix up this bug.
    // https://github.com/apache/incubator-mxnet/issues/12613
    //delete exec_;
    exec_ = nullptr;    
  }

  if (arg_name_idx_ != nullptr) {
    delete arg_name_idx_;
    arg_name_idx_ = nullptr;
  }
}

Status MxnetRunner::Init(const std::string &graph_def, 
                         const std::string& device_type) {
  device_type_ = device_type;
  loss_ = mxnet::cpp::Symbol::LoadJSON(graph_def);
  if (loss_.GetHandle() == nullptr) {
    return Status::Internal("load graph failed");
  }

  return Status::Ok();
}

Status MxnetRunner::Bind(mxnet::cpp::Executor *&exec,
                         const InputList& inputs) {
  using mxnet::cpp::NDArray;
  using mxnet::cpp::Context;
  using mxnet::cpp::Shape;
  std::map<std::string, NDArray> args;
  Context ctx = device_type_ == "cpu" ? Context::cpu() : Context::gpu();
  const std::vector<std::string> &arg_names = loss_.ListArguments();
  arg_sets_.insert(arg_names.begin(), arg_names.end());
  for (auto& item: inputs) {
    if (arg_sets_.find(item.first) != arg_sets_.end()) {
      mxnet::cpp::NDArray nd_array;
      XDL2MX::ConvertTensor(&ctx, item.second, &nd_array);
      args[item.first] = nd_array;
      //LOG(INFO) << "bind " << item.first << nd_array.GetData();
    }
  }

  std::map<std::string, NDArray> auxs;
  const std::vector<std::string> &aux_names = loss_.ListAuxiliaryStates();
  aux_sets_.insert(aux_names.begin(), aux_names.end());  
  for (auto& item: inputs) {
    if (aux_sets_.find(item.first) != aux_sets_.end()) {
      mxnet::cpp::NDArray nd_array;
      XDL2MX::ConvertTensor(&ctx, item.second, &nd_array);
      auxs[item.first] = nd_array;
    }
  }

  exec = loss_.SimpleBind(
      ctx, args, std::map<std::string, NDArray>(), 
      std::map<std::string, mxnet::cpp::OpReqType>(), auxs);
  const size_t arg_names_size = arg_names.size();
  const size_t inputs_size = inputs.size();
  assert(inputs_size >= arg_names_size);
  assert(arg_name_idx_ == nullptr);
  arg_name_idx_ = new size_t[inputs_size];
  for (size_t i = 0; i < inputs_size; ++i) {
    size_t idx;
    for (idx = 0; idx < arg_names_size; ++idx) {
      if (inputs[i].first == arg_names[idx]) {
        arg_name_idx_[i] = idx;
        break;
      }
    }

    if (idx >= arg_names_size) {
      arg_name_idx_[i] = unused_idx_;
    }
  }

  return Status::Ok();
}

bool MxnetRunner::Reshape(mxnet::cpp::Executor *&exec,
                          const InputList& inputs) {
  using mxnet::cpp::NDArray;
  using mxnet::cpp::OpReqType;
  using mxnet::cpp::Context;
  using mxnet::cpp::Shape;
  bool is_reshape = false;
  const size_t inputs_size = inputs.size();
  size_t arrays_size = inputs_size;
  Context ctx = device_type_ == "cpu" ? Context::cpu() : Context::gpu();
  for (size_t i = 0; i < inputs_size; ++i) {
    const size_t idx = arg_name_idx_[i];
    if (idx == unused_idx_) {
      --arrays_size;
      continue;
    }

    int compare = XDL2MX::CompareShape(inputs[i].second.Shape(), exec->arg_arrays[idx].GetShape());
    if (compare > 0) {
      XDL2MX::ConvertTensor(&ctx, inputs[i].second, &exec->arg_arrays[idx]);
      XDL2MX::ConvertTensor(&ctx, inputs[i].second, &exec->grad_arrays[idx]);
      is_reshape = true;
    } else if (compare < 0) {
      XDL2MX::ReshapeTensor(&ctx, inputs[i].second.Shape(), &exec->arg_arrays[idx]);
      XDL2MX::ReshapeTensor(&ctx, inputs[i].second.Shape(), &exec->grad_arrays[idx]);
      is_reshape = true;
    }
  }

  if (is_reshape) {
    std::vector<NDArray> &arg_arrays = exec->arg_arrays;
    std::vector<NDArray> &grad_arrays = exec->grad_arrays;
    std::vector<OpReqType> grad_reqs(arrays_size, OpReqType::kWriteTo);
    std::vector<NDArray> &aux_arrays = exec->aux_arrays;
    if (has_init_grad_) {
      *init_grad_ = grad_arrays;
    }
    mxnet::cpp::Executor *new_exec = loss_.Bind(ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
    delete exec;
    exec = new_exec;
  }

  return is_reshape;
}

Status MxnetRunner::Run(const InputList& inputs, 
                        DataList* outputs, 
                        DataList* gradients) {
  bool is_bind = false;
  if (exec_ == nullptr) {
    Status ret = Bind(exec_, inputs);
    if (ret != Status::Ok()) return ret;
    is_bind = true;
  }

  if (enable_reshape_ && !is_bind) {
    is_bind = Reshape(exec_, inputs);
  }

  auto arg_dict = exec_->arg_dict();
  auto aux_dict = exec_->aux_dict();

  //#pragma omp parallel for num_threads(2)
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& item = inputs[i];
    if (arg_sets_.find(item.first) != arg_sets_.end()) {
      XDL2MX::CopyTensor(item.second, &arg_dict[item.first]);
    } else if (aux_dict.find(item.first) != aux_dict.end()) {
      XDL2MX::CopyTensor(item.second, &aux_dict[item.first]);
    }
  }

  mxnet::cpp::NDArray::WaitAll();
  exec_->Forward(is_training_);
  if (is_training_) {
    if (has_init_grad_) {
      exec_->Backward(*init_grad_);
    } else {
      exec_->Backward();
    }

    auto grad_dict = exec_->grad_dict();
    for (auto& item: inputs) {
      auto iter = grad_dict.find(item.first);
      if (iter == grad_dict.end()) {
        continue;
      }
      gradients->push_back(iter->second);
    }
  }

  outputs->insert(outputs->end(), 
                  exec_->outputs.begin(), 
                  exec_->outputs.end());

  for (auto& item: *outputs) item.WaitToRead();
  for (auto& item: *gradients) item.WaitToRead();
  return Status::Ok();
}

#ifdef USE_GPU
Status MxnetRunner::Run(const InputList& inputs, 
                        DataList* outputs, 
                        DataList* gradients,
                        cudaStream_t stream) {
  bool is_bind = false;
  if (exec_ == nullptr) {
    Status ret = Bind(exec_, inputs);
    if (ret != Status::Ok()) return ret;
    is_bind = true;
  }

  if (enable_reshape_ && !is_bind) {
    is_bind = Reshape(exec_, inputs);
  }

  auto arg_dict = exec_->arg_dict();
  auto aux_dict = exec_->aux_dict();

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& item = inputs[i];
    if (arg_sets_.find(item.first) != arg_sets_.end()) {
      XDL2MX::CopyGpuTensorAsync(item.second, &arg_dict[item.first], stream);
    } else if (aux_dict.find(item.first) != aux_dict.end()) {
      XDL2MX::CopyGpuTensorAsync(item.second, &aux_dict[item.first], stream);
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));      
  exec_->Forward(is_training_);
  if (is_training_) {
    if (has_init_grad_) {
      exec_->Backward(*init_grad_);
    } else {
      exec_->Backward();
    }

    auto grad_dict = exec_->grad_dict();
    for (auto& item: inputs) {
      auto iter = grad_dict.find(item.first);
      if (iter == grad_dict.end()) {
        continue;
      }
      gradients->push_back(iter->second);
    }
  }

  outputs->insert(outputs->end(), 
                  exec_->outputs.begin(), 
                  exec_->outputs.end());

  for (auto& item: *outputs) item.WaitToRead();
  for (auto& item: *gradients) item.WaitToRead();
  return Status::Ok();
}

#endif

} // namespace xdl

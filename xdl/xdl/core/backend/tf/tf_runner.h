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

#ifndef XDL_BACKEND_TF_TF_RUNNER_H_
#define XDL_BACKEND_TF_TF_RUNNER_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "xdl/core/lib/status.h"
#include "xdl/core/backend/log_macro_undef.h"
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/tensor_shape.pb.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include "xdl/core/backend/log_macro_undef.h"

namespace xdl {

class TFRunner {
 public:
  using InputList = std::vector<std::pair<std::string, tensorflow::Tensor> >;
  TFRunner();
  ~TFRunner();

  Status Init(const std::string &graph_def_pb, float gpu_memory_fraction=0.5);
  Status Run(const InputList& inputs,
             const std::vector<std::string>& ops_names,
             std::vector<tensorflow::Tensor>* outputs);

 private:
  static const std::string TF_RUNNER_ROOT;
  tensorflow::Session *session_;
};

template <typename T>
tensorflow::Tensor MakeTensor(const std::vector<T>& data,
                              const std::vector<size_t>& dims) {
  tensorflow::TensorShape shape;
  for (size_t dim: dims) shape.AddDim(dim);
  tensorflow::Tensor ret(tensorflow::DataTypeToEnum<T>::value, shape);
  std::copy_n(data.data(), data.size(), ret.flat<T>().data());
  return ret;
}

} // namespace xdl

#endif // XDL_BACKEND_TF_TF_RUNNER_H_

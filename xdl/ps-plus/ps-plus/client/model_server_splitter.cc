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

#include "ps-plus/client/model_server_splitter.h"
#include "ps-plus/common/initializer/none_initializer.h"
#include <iostream>

namespace ps {
namespace client {

Status ModelServerSplitter::Init(int server_size, Tensor ids) {
  server_size_ = server_size;
  if (ids.Shape().Dims().size() != 1) {
    return Status::ArgumentError("Model Server Splitter: should be rank 1");
  }
  if (ids.Type() != DataType::kInt64) {
    return Status::ArgumentError("Model Server Splitter: should be Int64");
  }
  size_t size = ids.Shape().Dims()[0];
  int64_t* raw = ids.Raw<int64_t>();
  ids_.resize(size);
  server_ids_.resize(server_size_);
  for (int i = 0; i < server_size_; i++) {
    server_ids_[i].clear();
  }
  for (size_t i = 0; i < size; i++) {
    int k = (raw[i] & 1023) % server_size;
    ids_[i] = k;
    server_ids_[k].push_back(i);
  }
  return Status::Ok();
}

Status ModelServerSplitter::Split(Tensor t, std::vector<Tensor>* rst) {
  rst->resize(server_size_);
  std::vector<size_t> dims = t.Shape().Dims();
  size_t k = dims[0] == 0 ? 0 : t.Shape().NumElements() / dims[0];
  size_t block = SizeOfType(t.Type()) * k;
  if (dims.size() == 0 || dims[0] != ids_.size()) {
    return Status::ArgumentError("Model Server Splitter: Split Tensor dim[0] should equal to ids");
  }
  for (int i = 0; i < server_size_; i++) {
    dims[0] = server_ids_[i].size();
    (*rst)[i] = Tensor(t.Type(), TensorShape(dims), new initializer::NoneInitializer);
  }
  char* ptr = t.Raw<char>();
  std::vector<char*> rst_ptr(server_size_);
  for (int i = 0; i < server_size_; i++) {
    rst_ptr[i] = (*rst)[i].Raw<char>();
  }
  for (int k : ids_) {
    memcpy(rst_ptr[k], ptr, block);
    rst_ptr[k] += block;
    ptr += block;
  }
  return Status::Ok();
}

Status ModelServerSplitter::Combine(int id, Tensor data, Tensor* rst) {
  const std::vector<size_t>& dims = data.Shape().Dims();
  size_t k = dims[0] == 0 ? 0 : data.Shape().NumElements() / dims[0];
  size_t block = SizeOfType(data.Type()) * k;
  DataType type = data.Type();
  if (dims.size() == 0) {
    return Status::ArgumentError("Model Server Splitter: tensor shape error");
  }
  if (!rst->Initialized()) {
    std::vector<size_t> xdims = dims;
    xdims[0] = ids_.size();
    *rst = Tensor(type, TensorShape(xdims), new initializer::NoneInitializer);
  } else {
    const std::vector<size_t>& xdims = rst->Shape().Dims();
    if (data.Type() != rst->Type()) {
      return Status::ArgumentError("ModelServerSplitter: tensor type error");
    }
    if (dims.size() != xdims.size()) {
      return Status::ArgumentError("Model Server Splitter: tensor shape error");
    }
    if (dims[0] != server_ids_[id].size()) {
      return Status::ArgumentError("Model Server Splitter: tensor shape error");
    }
    for (size_t j = 1; j < dims.size(); j++) {
      if (dims[j] != xdims[j]) {
        return Status::ArgumentError("Model Server Splitter: tensor shape error");
      }
    }
  }
  char* data_ptr = data.Raw<char>();
  char* ptr = rst->Raw<char>();
  for (auto item : server_ids_[id]) {
    memcpy(ptr + item * block, data_ptr, block);
    data_ptr += block;
  }
  return Status::Ok();
}

}
}


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

#include "ps-plus/common/tensor.h"
#include <cstring>

namespace ps {

Tensor::Tensor() : state_(nullptr) {}

Tensor::Tensor(DataType type, const TensorShape& shape, Initializer* initializer, bool init)
  : state_(new State(
        new char[SizeOfType(type) * shape.NumElements()],
        type, shape, initializer, true)) {
  if (init) {
    state_->initializer->MultiThreadInit(state_->buffer, state_->type, state_->shape.NumElements());
  }
}

Tensor::Tensor(DataType type, TensorShape&& shape, Initializer* initializer, bool init)
  : state_(new State(
        new char[SizeOfType(type) * shape.NumElements()],
        type, std::move(shape), initializer, true)) {
  if (init) {
    state_->initializer->MultiThreadInit(state_->buffer, state_->type, state_->shape.NumElements());
  }
}

Tensor::Tensor(DataType type, const TensorShape& shape, char* buffer, Initializer* initializer)
  : state_(new State(buffer, type, shape, initializer, false)) {}

Tensor::Tensor(DataType type, TensorShape&& shape, char* buffer, Initializer* initializer)
  : state_(new State(buffer, type, std::move(shape), initializer, false)) {}

Tensor::Tensor(const Tensor& rhs) : state_(rhs.state_) {
  Ref();
}

Tensor::Tensor(Tensor&& rhs) : state_(rhs.state_) {
  rhs.state_ = nullptr;
}

Tensor::~Tensor() {
  UnRef();
}

Tensor& Tensor::operator=(const Tensor& rhs) {
  UnRef();
  state_ = rhs.state_;
  Ref();
  return *this;
}

Tensor& Tensor::operator=(Tensor&& rhs) {
  std::swap(state_, rhs.state_);
  return *this;
}

void Tensor::ReShape(const TensorShape& shape) {
  size_t old_size = state_->shape.NumElements() * SizeOfType(state_->type);
  size_t new_size = shape.NumElements() * SizeOfType(state_->type);

  State* new_state = new State(new char[new_size], state_->type, shape, state_->initializer->Clone(), true);
  if (new_size <= old_size) {
    QuickMemcpy(new_state->buffer, state_->buffer, new_size);
  } else {
    QuickMemcpy(new_state->buffer, state_->buffer, old_size);
    new_state->initializer->MultiThreadInit(new_state->buffer + old_size, new_state->type, new_state->shape.NumElements() - state_->shape.NumElements());
  }
  UnRef();
  state_ = new_state;
}

void Tensor::Clear(size_t beg, size_t size) {
  state_->initializer->MultiThreadInit(state_->buffer + beg * SizeOfType(state_->type), state_->type, size);
}

void Tensor::ClearId(size_t id) {
  size_t size = state_->shape.NumElements() / state_->shape[0];
  Clear(id * size, size);
}

Tensor Tensor::Clone() {
  if (state_ == nullptr) {
    return Tensor();
  }
  Tensor ret(state_->type, state_->shape, state_->initializer->Clone(), false);
  memcpy(ret.state_->buffer, state_->buffer, state_->shape.NumElements() * SizeOfType(state_->type));
  return ret;
}

void Tensor::UnRef() {
  if (state_ == nullptr) {
    return;
  }
  if (--state_->ref == 0) {
    if (state_->own_buffer) {
      delete [] state_->buffer;
    }
    delete state_;
    state_ = nullptr;
  }
}

void Tensor::Ref() {
  if (state_ == nullptr) {
    return;
  }
  state_->ref++;
}

}


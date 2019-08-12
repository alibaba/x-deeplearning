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
#include "ps-plus/common/logging.h"
#include <cstring>

namespace ps {

Tensor::Tensor(TType tensor_type) : state_(nullptr), tensor_type_(tensor_type) {}

Tensor::Tensor(DataType type, const TensorShape& shape, Initializer* initializer, TType tensor_type, bool init) {
  tensor_type_ = tensor_type;
  if (tensor_type == TType::kContinuous) {
    state_ = new ContinuousState(new char[SizeOfType(type) * shape.NumElements()], type, shape, initializer, true, init);
  } else {
    state_ = new SegmentState(type, shape, initializer, init, DEFAULT_SEGMENT_SIZE);
  }
}

Tensor::Tensor(DataType type, const TensorShape& shape, char* buffer, Initializer* initializer)
  : state_(new ContinuousState(buffer, type, shape, initializer, false, false)), tensor_type_(TType::kContinuous) {}

Tensor::Tensor(DataType type, const TensorShape& shape, Initializer* initializer, bool init, size_t segment_size)
  : state_(new SegmentState(type, shape, initializer, init, segment_size)), tensor_type_(TType::kSegment) {}

Tensor::Tensor(const Tensor& rhs) : state_(rhs.state_), tensor_type_(rhs.tensor_type_) {
  Ref();
}

Tensor::Tensor(Tensor&& rhs) : state_(rhs.state_), tensor_type_(rhs.tensor_type_) {
  rhs.state_ = nullptr;
}

Tensor::~Tensor() {
  UnRef();
}

Tensor& Tensor::operator=(const Tensor& rhs) {
  UnRef();
  state_ = rhs.state_;
  tensor_type_ = rhs.tensor_type_;
  Ref();
  return *this;
}

Tensor& Tensor::operator=(Tensor&& rhs) {
  std::swap(state_, rhs.state_);
  std::swap(tensor_type_, rhs.tensor_type_);
  return *this;
}

Status Tensor::InitChunkFrom(const size_t& start_index) {
  if (tensor_type_ == TType::kContinuous) {
    return Status::NotImplemented("ContinuousTensor can't support InitChunkFrom function");
  }
  SegmentState* state = dynamic_cast<SegmentState*>(state_);
  size_t trunk_start = start_index % state->segment_size;
  if (trunk_start == 0) {
    return Status::Ok();
  }
  void* ptr = state->Raw(start_index);
  state->initializer->MultiThreadInit(ptr, state->type, (state->segment_size - trunk_start) * state->slice_size);
  //LOG_INFO("Call InitChunkFrom in tensor, start %ld, trunk_start %ld, size %ld", start_index, trunk_start, (state->segment_size - trunk_start) * state->slice_size);
  return Status::Ok();
}

void Tensor::ReShape(const TensorShape& shape) {
  if (tensor_type_ == TType::kContinuous) {
    size_t old_size = state_->shape.NumElements() * SizeOfType(state_->type);
    size_t new_size = shape.NumElements() * SizeOfType(state_->type);
    ContinuousState* new_state = new ContinuousState(new char[new_size], state_->type, shape, state_->initializer->Clone(), true, false);
    ContinuousState* old_state = dynamic_cast<ContinuousState*>(state_);
    if (new_size <= old_size) {
      QuickMemcpy(new_state->buffer, old_state->buffer, new_size);
    } else {
      QuickMemcpy(new_state->buffer, old_state->buffer, old_size);
      new_state->initializer->MultiThreadInit(new_state->buffer + old_size, new_state->type, new_state->shape.NumElements() - state_->shape.NumElements());
    }
    UnRef();
    state_ = new_state;
  } else {
    SegmentState* state = dynamic_cast<SegmentState*>(state_);
    if (state == nullptr) {
      std::cerr << "state_ for segment tensor is nullptr\n";
      abort();
    }
    size_t index = shape[0]/state->segment_size;
    if (state->buffers.size() > index && state->buffers[index] != nullptr) {
      return;
    }
    while (state->buffers.size() <= index) {
      char* ptr = new char[state->chunk_size];
      state->initializer->MultiThreadInit(ptr, state->type, state->slice_size * state->segment_size);
      state->buffers.push_back(ptr);
    }
    while (state->buffers[index] == nullptr) {
    }
    state->shape.Set(0, state->buffers.size() * state->segment_size);
  }
}

void Tensor::ClearId(size_t id) {
  SegmentState* state = dynamic_cast<SegmentState*>(state_);
  state_->initializer->MultiThreadInit(state->Raw(id), state->type, state->slice_size);
}

size_t Tensor::SegmentSize() const {
  SegmentState* state = dynamic_cast<SegmentState*>(state_);
  if (state == nullptr) {
    std::cerr << "Only Segment tensor can call SegmentSize\n";
    abort();
  }
  return state->segment_size;
}

void Tensor::SetOwnBuffer(bool own) {
  ContinuousState* state = dynamic_cast<ContinuousState*>(state_);
  if (state == nullptr) {
    std::cerr << "Only Continuous tensor can call SetOwnBuffer\n";
    abort();
  }
  state->own_buffer = own;
}

Tensor Tensor::Clone() const {
  if (state_ == nullptr) {
    return Tensor();
  }
  Initializer* init = nullptr;
  if (state_->initializer != nullptr) {
    init = state_->initializer->Clone();
  }
  if (tensor_type_ == TType::kContinuous) {
    Tensor ret(state_->type, state_->shape, init, TType::kContinuous, false);
    ContinuousState* this_state = dynamic_cast<ContinuousState*>(state_);
    ContinuousState* ret_state = dynamic_cast<ContinuousState*>(ret.state_);
    memcpy(ret_state->buffer, this_state->buffer, state_->shape.NumElements() * SizeOfType(state_->type));
    return ret;
  } else {
    Tensor ret(state_->type, state_->shape, init, TType::kSegment, false);
    SegmentState* this_state = dynamic_cast<SegmentState*>(state_);
    SegmentState* ret_state = dynamic_cast<SegmentState*>(ret.state_);
    for (size_t i = 0; i < this_state->buffers.size(); i++) {
      memcpy(ret_state->buffers[i], this_state->buffers[i], this_state->chunk_size);
    }
    return ret;
  }
}

void Tensor::UnRef() {
  if (state_ == nullptr) {
    return;
  }
  if (--state_->ref == 0) {
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

const int64_t Tensor::DEFAULT_SEGMENT_SIZE = 1 << 12;

}


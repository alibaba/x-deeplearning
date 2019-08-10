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

#ifndef PS_PLUS_COMMON_TENSOR_H_
#define PS_PLUS_COMMON_TENSOR_H_

#include <memory>
#include <atomic>
#include <iostream>
#include "tbb/parallel_for.h"
#include "tbb/concurrent_vector.h"

#include "ps-plus/common/types.h"
#include "ps-plus/common/tensor_shape.h"
#include "ps-plus/common/initializer.h"
#include "ps-plus/common/thread_pool.h"

namespace ps {

class Tensor {
 public:
  enum TType {
    kContinuous,
    kSegment,
  };
  Tensor(TType tensor_type = TType::kContinuous);
  Tensor(DataType type, const TensorShape& shape, Initializer* initializer, TType tensor_type = TType::kContinuous, bool init = true);
  // used for only kContinuous tensor
  Tensor(DataType type, const TensorShape& shape, char* buffer, Initializer* initializer);
  // used for only kSegment tensor  
  Tensor(DataType type, const TensorShape& shape, Initializer* initializer, bool init, size_t segment_size);
  
  Tensor(const Tensor& rhs);
  Tensor(Tensor&& rhs);
  ~Tensor();

  Tensor& operator=(const Tensor& rhs);
  Tensor& operator=(Tensor&& rhs);

  bool Initialized() const { return state_ != nullptr; }
  DataType Type() const { return state_->type; }
  Initializer* GetInitializer() const { return state_->initializer.get(); }
  const TensorShape& Shape() const { return state_->shape; }
  TType TensorType() const {return tensor_type_;}
  void SetInititalizer(Initializer* init) {state_->initializer.reset(init);}
  Status InitChunkFrom(const size_t& start_index);

  // Note: We don't check the type_. Everyone who call following method
  // should use CASES or just check the type_;
  // Not safe to use for Segment Tensor;
  template<typename T>
  T* Raw() const {
    return reinterpret_cast<T*>(state_->Raw(0));
  }

  template<typename T>
  T* Raw(size_t id) const {
    return reinterpret_cast<T*>(state_->Raw(id));
  }

  void ReShape(const TensorShape& shape);
  // Note: We don't check id
  void ClearId(size_t id);
  Tensor Clone() const;

  size_t SegmentSize() const;
  void SetOwnBuffer(bool own);
  const static int64_t DEFAULT_SEGMENT_SIZE;
 private:
  void UnRef();
  void Ref();

  struct State {
    State(DataType type_, const TensorShape& shape_, Initializer* initializer_)
      : type(type_), shape(shape_), initializer(initializer_), ref(1) {
    }
    virtual void* Raw(size_t id) = 0;
    DataType type;
    TensorShape shape;
    std::unique_ptr<Initializer> initializer;
    std::atomic<size_t> ref;
    virtual ~State() {}
  };

  struct ContinuousState: public State {
    ContinuousState(char* buffer_, DataType type_, const TensorShape& shape_, Initializer* initializer_, bool own_buffer_, bool init_)
      : State(type_, shape_, initializer_), buffer(buffer_), own_buffer(own_buffer_) {
      if (init_) {
        initializer->MultiThreadInit(buffer, type, shape.NumElements());
      }
    }
    virtual void* Raw(size_t id) {
      if (id == 0) {
        return buffer;
      }
      if (shape.IsScalar()) {
        return nullptr;
      }
      return buffer + id * shape.NumElements() / shape[0] * SizeOfType(type);
    }
    virtual ~ContinuousState() {
      if (own_buffer) {
        delete [] buffer;
        buffer = nullptr;
      }
    }
    bool own_buffer;
    char* buffer;
  };

  struct SegmentState: public State {
    SegmentState(DataType type_, const TensorShape& shape_, Initializer* initializer_, bool init_, size_t segment_size_)
      : State(type_, shape_, initializer_), segment_size(segment_size_) {
      if (shape_.IsScalar()) {
        throw std::invalid_argument("SegmentState don't allow scalar variable");
      }
      slice_size = shape_.NumElements() / shape_[0];
      chunk_size = segment_size_ * SizeOfType(type_) * slice_size;
      buffers.grow_to_at_least(shape_[0]/segment_size + (shape_[0] % segment_size == 0 ? 0 : 1), nullptr);
      for (size_t i = 0; i < buffers.size(); i++) {
        buffers[i] = new char[chunk_size];
      }
      if (init_) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, buffers.size() - 1), [&](tbb::blocked_range<size_t>& r) {
              for (size_t i = r.begin(); i < r.end(); i++) {
                initializer_->MultiThreadInit(buffers[i], type_, segment_size * slice_size);
              }
            });
      }
      // 因为我们有预留的空间，所以最后一个buffer必须初始化
      if (initializer_ != nullptr) {
        initializer_->MultiThreadInit(buffers[buffers.size()-1], type_, segment_size * slice_size);
      }
      shape.Set(0, buffers.size() * segment_size);
    }
    virtual ~SegmentState() {
      for (size_t i = 0; i < buffers.size(); i++) {
        delete [] buffers[i];
      }
    }
    virtual void* Raw(size_t id) {
      return buffers[id/segment_size] + (id%segment_size) * slice_size * SizeOfType(type);
    }
    size_t segment_size;
    size_t chunk_size;
    size_t slice_size;
    tbb::concurrent_vector<char*> buffers;
  };
  State* state_;
  TType tensor_type_;
};

}

#endif


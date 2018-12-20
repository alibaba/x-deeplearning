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

#include "ps-plus/common/types.h"
#include "ps-plus/common/tensor_shape.h"
#include "ps-plus/common/initializer.h"
#include "ps-plus/common/thread_pool.h"

#include <memory>
#include <atomic>

namespace ps {

class Tensor {
 public:
  Tensor();

  Tensor(DataType type, const TensorShape& shape, Initializer* initializer, bool init = true);

  Tensor(DataType type, TensorShape&& shape, Initializer* initializer, bool init = true);

  Tensor(DataType type, const TensorShape& shape, char* buffer, Initializer* initializer);

  Tensor(DataType type, TensorShape&& shape, char* buffer, Initializer* initializer);

  Tensor(const Tensor& rhs);

  Tensor(Tensor&& rhs);

  ~Tensor();

  Tensor& operator=(const Tensor& rhs);

  Tensor& operator=(Tensor&& rhs);

  bool Initialized() { return state_ != nullptr; }
  DataType Type() const { return state_->type; }
  const TensorShape& Shape() const { return state_->shape; }
  Initializer* GetInitializer() const { return state_->initializer.get(); }

  // Note: We don't check the type_. Everyone who call following method
  // should use CASES or just check the type_;
  template<typename T>
  T* Raw() const {
    return reinterpret_cast<T*>(state_->buffer);
  }

  void ReShape(const TensorShape& shape);

  // Note: We don't check the beg and size
  void Clear(size_t beg, size_t size);

  // Note: We don't check id
  void ClearId(size_t id);

  Tensor Clone();
 private:
  void UnRef();
  void Ref();

  struct State {
    State(char* buffer_, DataType type_, const TensorShape& shape_, Initializer* initializer_, bool own_buffer_)
      : buffer(buffer_), type(type_), shape(shape_), initializer(initializer_), own_buffer(own_buffer_), ref(1) {}
    State(char* buffer_, DataType type_, TensorShape&& shape_, Initializer* initializer_, bool own_buffer_)
      : buffer(buffer_), type(type_), shape(std::move(shape_)), initializer(initializer_), own_buffer(own_buffer_), ref(1) {}
    char* buffer;
    DataType type;
    TensorShape shape;
    std::unique_ptr<Initializer> initializer;
    bool own_buffer;
    std::atomic<size_t> ref;
  };
  State* state_;
};

}

#endif


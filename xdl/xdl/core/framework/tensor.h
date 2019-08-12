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

#ifndef XDL_CORE_FRAMEWORK_TENSOR_H_
#define XDL_CORE_FRAMEWORK_TENSOR_H_

#include "xdl/core/utils/logging.h"
#include "xdl/core/lib/refcount.h"
#include "xdl/core/framework/device.h"
#include "xdl/core/framework/types.h"
#include "xdl/core/framework/tensor_shape.h"

namespace xdl {

class Buffer : public RefCounted {
 public:
  Buffer(Allocator* allocator, size_t size);
  Buffer(Allocator* allocator, void* begin, size_t size, bool own);
  Buffer(void* begin, size_t size, Buffer* parent);
  ~Buffer();

  void* begin() {
    return begin_;
  }

  size_t size() {
    return size_;
  }

  Device* device() {
    return device_;
  }
 private:
  RefCountedPtr<Allocator> allocator_;
  Device* device_;
  void* begin_;
  size_t size_;
  bool own_;
  RefCountedPtr<Buffer> parent_;
};

class Tensor {
 public:
  Tensor();
  Tensor(Device* device, const TensorShape& shape, DataType type);
  Tensor(Device* device, const TensorShape& shape, DataType type,
         void* data, bool own);
  Tensor(Allocator* allocator, const TensorShape& shape, DataType type);
  Tensor(Allocator* allocator, const TensorShape& shape, DataType type,
         void* data, bool own);
  Tensor(const TensorShape& shape, DataType type,
         Buffer* buffer);

  bool Initialized() const {
    return state_.get() != nullptr;
  }

  template <typename T>
  T* Raw() const {
    XDL_CHECK(Initialized()) << "Tensor Not Initialized";
    return reinterpret_cast<T*>(state_->buffer->begin());
  }

  TensorShape Shape() const {
    XDL_CHECK(Initialized());
    return state_->shape;
  }

  DataType Type() const {
    XDL_CHECK(Initialized());
    return state_->type;
  }

  template <typename T>
  T Scalar() const {
    XDL_CHECK(Shape().IsScalar()) << "tensor is not scala";
    return *(Raw<T>());
  }

  Buffer* GetBuffer() const {
    XDL_CHECK(Initialized());
    return state_->buffer.get();
  }

 private:
  class State : public RefCounted {
   public:
    State(const RefCountedPtr<Buffer>& buffer_,
          const TensorShape& shape_, DataType type_)
      : buffer(buffer_), shape(shape_), type(type_) {}
    RefCountedPtr<Buffer> buffer;
    TensorShape shape;
    DataType type;
  };
  RefCountedPtr<State> state_;
};

template <>
inline std::string Tensor::Scalar<std::string>() const {
  if (Shape().Size() == 0) {
    return "";
  }

  return std::string(reinterpret_cast<char*>(Raw<int8_t>()), 
                     Shape().NumElements());
}

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_TENSOR_H_


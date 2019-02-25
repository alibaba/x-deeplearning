/*
 * \file blob.h 
 * \brief The blob definition
 */
#pragma once

#include "blaze/common/allocator.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/log.h"
#include "blaze/common/types.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

// typedef size_t TIndex;

class Blob {
 public:
  explicit Blob(const DeviceOption& device_option) :
      device_option_(device_option) { }
  virtual ~Blob();

  explicit Blob(const DeviceOption& device_option, const std::vector<TIndex>& dims, DataType data_type) :
      device_option_(device_option),
      data_type_(data_type) {
    Reshape(dims);
  }

  inline void Release() {
    Destroy();
    size_ = capacity_ = 0;
    dims_.clear();
  }

  inline void Reshape(const std::vector<TIndex>& dims) {
    TIndex new_size = 1;
    for (auto d : dims) {
      new_size *= d;
    }
    size_ = new_size;
    if (size_ > capacity_) {
      const int device_id = device_option_.device_id();
      const int device_type = device_option_.device_type();
      blaze::Free(data_, capacity_ * DataTypeSize(data_type_), device_type, device_id);
      capacity_ = size_;
      data_ = blaze::Alloc(capacity_ * DataTypeSize(data_type_), device_type, device_id);
    }
    dims_ = dims;
  }

  inline void RefReshape(const std::vector<TIndex>& dims, void* handle) {
    Destroy();

    own_handle_ = false;
    TIndex new_size = 1;
    for (auto d : dims) {
      new_size *= d;
    }
    size_ = new_size;
    capacity_ = size_;
    dims_ = dims;
    data_ = handle;
  }

  inline const DeviceOption& device_option() const { return device_option_; }
  inline int device_type() const { return device_option_.device_type(); }
  inline void set_data_type(DataType data_type) { data_type_ = data_type; }
  inline int data_type() const { return data_type_; }
  inline int device_id() const { return device_option_.device_id(); }

  inline std::vector<TIndex>& shape() { return dims_; }
  inline const std::vector<TIndex>& shape() const { return dims_; }

  inline TIndex capacity() const { return capacity_; }
  inline TIndex size() const { return size_; }
  inline TIndex size(int start, int end) {
    TIndex s = 1;
    for (int k = start; k < end; ++k) {
      s *= dims_[k];
    }
    return s;
  }
  inline TIndex dim(size_t index) {
    BLAZE_CONDITION_THROW(index < dims_.size(), "index=", index, "dims_.size()=", dims_.size());
    return dims_[index];
  }

  template <typename T>
  inline T* as() { return reinterpret_cast<T*>(data_); }
  inline void* data() { return data_; }
  inline const void* data() const { return data_; }

 protected:
  void Destroy();

  DeviceOption device_option_;
  void* data_ = nullptr;
  TIndex capacity_ = 0;
  TIndex size_ = 0;
  std::vector<TIndex> dims_;
  int data_type_ = DataType::kFloat;
  bool own_handle_ = true;
};

// stream_id is zero on cpu, cudaStream_t on gpu.
extern void Copy(Blob* dst, const Blob* src, void* stream);

}  // namespace blaze

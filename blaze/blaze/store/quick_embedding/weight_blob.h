/*!
 * \file weight_blob.h
 * \desc Sparse weight blob
 */
#pragma once

#include <malloc.h>

#include "blaze/store/quick_embedding/serializable.h"
#include "blaze/common/log.h"

namespace blaze {
namespace store {

template<typename T>
class WeightBlob : public Serializable {
 public:
  WeightBlob() :
      bytes_(nullptr),
      byte_size_(1),
      capacity_size_(0) {}

  ~WeightBlob() override {
    if (bytes_)
      free(bytes_);
  }

  bool AllocateMemory(size_t size) {
    size_t len = sizeof(T) * size + 1;
    char *new_bytes = reinterpret_cast<char *>(malloc(len));
    if (!new_bytes) {
      LOG_ERROR("bad alloc memory of whole blob!");
      return false;
    }
    if (bytes_)
      free(bytes_);
    bytes_ = new_bytes;
    capacity_size_ = len;
    return true;
  }

  uint64_t InsertWeights(int dim, T **weights) {
    size_t len = sizeof(T) * dim;
    if (byte_size_ + len > capacity_size_) {
      return 0;
    }

    uint64_t ret = byte_size_;
    *weights = (T *) (bytes_ + byte_size_);
    byte_size_ += len;
    return ret;
  }

  T *GetWeights(uint64_t offset) const {
    if (offset >= byte_size_) {
      return nullptr;
    }
    return (T *) (bytes_ + offset);
  }

  uint64_t ByteArraySize() const override {
    return sizeof(byte_size_) + byte_size_;
  }

  bool Load(std::istream *is) override {
    // [step1]: load byte size
    is->read((char *) &byte_size_, sizeof(byte_size_));
    if (!is->good()) return false;
    capacity_size_ = byte_size_;
    // [step2]: reallocate memory
    char *new_bytes = reinterpret_cast<char *>(malloc(byte_size_));
    if (!new_bytes) {
      LOG_ERROR("bad allocate memory of weight blob while loading");
      return false;
    }
    if (bytes_)
      free(bytes_);
    bytes_ = new_bytes;
    // [step3]: load bytes
    is->read(bytes_, byte_size_);
    if (!is->good()) return false;

    return true;
  }

  bool Dump(std::ostream *os) const override {
    if (byte_size_ == 0 || bytes_ == nullptr)
      return false;

    // [step1]: write byte size
    os->write((char *) &byte_size_, sizeof(byte_size_));
    if (!os->good()) return false;
    // [ste2]: write bytes
    os->write(bytes_, byte_size_);
    if (!os->good()) return false;

    return true;
  }

 private:
  char *bytes_;
  uint64_t byte_size_;
  uint64_t capacity_size_;
};

}  // namespace store
}  // namespace blaze



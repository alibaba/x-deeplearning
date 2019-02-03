/*
 * \file allocator.h 
 * \brief The allocator
 */
#pragma once

#include <mutex>

#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

#ifdef USE_CUDA
class CUDADeviceGuard {
 public:
  explicit CUDADeviceGuard(int newdevice) {
    cudaGetDevice(&previous_);
    if (previous_ != newdevice) {
      cudaSetDevice(newdevice);
    }
  }
  ~CUDADeviceGuard() noexcept {
    cudaSetDevice(previous_);
  }

 private:
  int previous_;
};
#endif

template <int device_type>
struct Allocator {
  static void* Alloc(size_t size, int device_id) { return nullptr; }
  static void Free(void* data, size_t size, int device_id) { }
};

template <>
struct Allocator<kCPU> {
  static void* Alloc(size_t size, int device_id = 0) {
    void* data = nullptr;
    posix_memalign(&data, 64, size);
    return data;
  }
  static void Free(void* data, size_t size, int device_id = 0) {
    free(data);
  }
};

template <>
struct Allocator<kCUDA> {
  static void* Alloc(size_t size, int device_id) {
#ifdef USE_CUDA
    CUDADeviceGuard guard(device_id);
    void* data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, size));
    return data;
#else
    BLAZE_CONDITION_THROW("USE_CUDA is not enabled!");
#endif
  }
  static void Free(void* data, size_t size, int device_id) {
#ifdef USE_CUDA
    CUDADeviceGuard guard(device_id);
    (cudaFree(data));
#else
    BLAZE_CONDITION_THROW("USE_CUDA is not enabled!");
#endif
  }
};

// The slab allocator for blaze.
template <int device_type, int device_id>
class SlabAllocator {
 public:
  static SlabAllocator<device_type, device_id>* Get() {
    static std::shared_ptr<SlabAllocator<device_type, device_id>> inst(new SlabAllocator<device_type, device_id>);
    return inst.get();
  }
  inline void* Alloc(size_t size) {
    auto index = Index(size);
    std::unique_lock<std::mutex> lock(mutex_);
    if (!slab_list_[index].empty()) {
      auto ret = slab_list_[index].back();
      slab_list_[index].pop_back();
      return ret;
    } else {
      return AllocFromChunk(1 << index);
    }
  }
  inline void Free(void* data, size_t size) {
    auto index = Index(size);
    std::unique_lock<std::mutex> lock(mutex_);
    slab_list_[index].push_back(data);
  }

  virtual ~SlabAllocator() {
    if (free_chunk_) used_chunk_.push_back(free_chunk_);
    for (auto ptr : used_chunk_) {
      Allocator<device_type>::Free(ptr, kChunkSize, device_id);
    }
  }

 protected:
  SlabAllocator() : free_chunk_(nullptr) { AllocNewChunk(); }

  inline size_t Index(size_t size) {
    auto index = 1;
    auto space = 1 << index;
    while (space < size) {
      space = 1 << (++index);
    }
    return index;
  }
  inline void* AllocFromChunk(size_t size) {
    if (remaining_free_chunk_size_ < size) {
      AllocNewChunk();
    }
    BLAZE_CONDITION_THROW(
        remaining_free_chunk_size_ >= size,
        "remaining_free_chunk_size_=", remaining_free_chunk_size_,
        " size=", size);
    auto ret = static_cast<char*>(free_chunk_) + kChunkSize - remaining_free_chunk_size_;
    remaining_free_chunk_size_ -= size;
    return ret;
  }
  inline void AllocNewChunk() {
    if (free_chunk_) used_chunk_.push_back(free_chunk_);
    remaining_free_chunk_size_ = kChunkSize;
    free_chunk_ = Allocator<device_type>::Alloc(remaining_free_chunk_size_, device_id);
  }

  // Maximum allocated space is 256MB, 2^28
  static const int kMaxSlabNum = 28;
  std::vector<void*> slab_list_[kMaxSlabNum + 1];

  static const int kChunkSize = 1024 * 1024 * 1024; // 1GB
  std::vector<void*> used_chunk_;
  void* free_chunk_;
  size_t remaining_free_chunk_size_;

  std::mutex mutex_;
};

inline void* Alloc(size_t size, int device_type, int device_id) {
  switch (device_type) {
    case kCPU:
      return Allocator<kCPU>::Alloc(size, device_id);
    case kCUDA:
      return Allocator<kCUDA>::Alloc(size, device_id);
    default:
      BLAZE_CONDITION_THROW("Unnkown device_type=", device_type);
      return nullptr;
  }
}

inline void Free(void* data, size_t size, int device_type, int device_id) {
  switch (device_type) {
    case kCPU:
      Allocator<kCPU>::Free(data, size, device_id);
      break;
    case kCUDA:
      Allocator<kCUDA>::Free(data, size, device_id);
      break;
    default:
      BLAZE_CONDITION_THROW("Unnkown device_type=", device_type);
      break;
  }
}

}  // namespace blaze


/*
 * \file common_defines.h 
 * \brief The common defines
 */
#pragma once

#include <string.h>

#include <exception>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>

#include "blaze/common/log.h"

#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <curand.h>
#include <nvrtc.h>
#include <cudnn.h>

// CUDA: use 512 threads per block
#define CUDA_NUM_THREADS 512
// CUDA: number of blocks for threads
#define CUDA_GET_BLOCKS(N, Threads) (N + Threads - 1) / Threads
// CUDA: grid stride loop, only one grid.
#define CUDA_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
// CUDA: X dimension loop
#define CUDA_KERNEL_X_LOOP(i, n) CUDA_KERNEL_LOOP(i, n)
// CUDA: Y dimension loop
#define CUDA_KERNEL_Y_LOOP(i, n) \
  for (size_t i = blockIdx.y * blockDim.y + threadIdx.y; i < (n); i += blockDim.y * gridDim.y)

// CUDA: various check for different function calls
#ifndef CUDA_CHECK
#define CUDA_CHECK(condition) \
    do { \
      cudaError_t error = (condition); \
      if (error != cudaSuccess) { \
        BLAZE_THROW("cuda error status=", error, " msg=", cudaGetErrorString(error)); \
      } \
    } while (0)
#endif

#ifndef CURAND_CHECK
#define CURAND_CHECK(condition) \
    do { \
      curandStatus_t status = condition; \
      if (status != CURAND_STATUS_SUCCESS) { \
        BLAZE_THROW("cuda error status=", status); \
      } \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(condition) \
    do { \
      cublasStatus_t status = condition; \
      if (status != CUBLAS_STATUS_SUCCESS) { \
        BLAZE_THROW("cudablas error status=", status); \
      } \
    } while (0)
#endif

#ifndef NVRTC_CHECK
#define NVRTC_CHECK(condition) \
    do { \
      nvrtcResult status = condition; \
      if (status != NVRTC_SUCCESS) { \
        BLAZE_THROW("nvrtc error status=", status, " msg=", nvrtcGetErrorString(status)); \
      } \
    } while (0)
#endif

#ifndef CUDA_DRIVERAPI_CHECK
#define CUDA_DRIVERAPI_CHECK(condition) \
    do { \
      CUresult result = condition; \
      if (result != CUDA_SUCCESS) { \
        const char* msg; \
        cuGetErrorName(result, &msg); \
        BLAZE_THROW("cuda dri error=", msg); \
      } \
    } while (0)
#endif

#ifndef CUDNN_CHECK
#define CUDNN_CHECK(condition)         \
    do { \
      cudnnStatus_t status = condition; \
      if (status != CUDNN_STATUS_SUCCESS) { \
        BLAZE_THROW("cudnn error status=", status, " msg=", cudnnGetErrorString(status)); \
      } \
    } while (0)
#endif

#endif

#ifdef __CUDACC__
#define BLAZE_DEVICE __device__
#define BLAZE_HOST   __host__
#define BLAZE_INLINE __inline__
#define BLAZE_INLINE_X __inline__  __host__ __device__
#else
#define BLAZE_DEVICE
#define BLAZE_HOST
#define BLAZE_INLINE __inline__
#define BLAZE_INLINE_X BLAZE_INLINE
#endif

#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)           \
    classname(const classname&) = delete;            \
    classname& operator=(const classname&) = delete
#endif

#define CONCATENATE_IMPL(s1, s2)  s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1, s2)
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)

#define UNUSED __attribute__((__unused__))
#define USED __attribute__((__used__))

#ifdef __CUDACC__
#define ALIGNED(x) __align__(x)
#else
#define ALIGNED(x) __attribute__((aligned(x)))
#endif

#define AlignN(N, align) ((((N) + align - 1) / align) * align)

#define BLAZE_VERSION_MAJOR 2
#define BLAZE_VERSION_MINOR 0

namespace blaze {

// make_unique is a C++14 feature.
#if __cplusplus >= 201402L || (defined __cpp_lib_make_unique && __cpp_lib_make_unique >= 201304L)
using std::make_unique;
#else
template <typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
template <typename T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(const size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}
template <typename T, typename... Args>
typename std::enable_if<std::extent<T>::value != 0, std::unique_ptr<T>>::type
make_unique(Args&&...) = delete;
#endif

template <typename Key, typename Value>
using BlazeMap = std::map<Key, Value>;

typedef size_t TIndex;

template <typename T>
struct VectorDataGuard {
  virtual ~VectorDataGuard() {
    for (auto item : data) {
      delete item;
    }
  }
  std::vector<T*> data;
};

}  // namespace blaze

#ifdef USE_CUDA
#include "blaze/common/cuda_helpers.h"
#endif

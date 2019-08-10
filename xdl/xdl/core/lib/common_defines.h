/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef XDL_CORE_LIB_COMMON_DEFINES_H_
#define XDL_CORE_LIB_COMMON_DEFINES_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "xdl/core/utils/logging.h"

#ifdef USE_GPU
#include <driver_types.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#endif

/// helper macro t supress unused warning
#define ATTRIBUTE_UNUSED __attribute__((unused))

/// helper macro to generate string concat
#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) XDL_STR_CONCAT_(__x, __y)

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(T) \
    T(T const&) = delete; \
    T(T&&) = delete; \
    T& operator=(T const&) = delete; \
    T& operator=(T&&) = delete
#endif

#ifdef USE_GPU
// CUDA: use 512 threads per block
#define CUDA_NUM_THREADS 1024
// CUDA: number of blocks
#define CUDA_GET_BLOCKS(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)
// CUDA: number of threads in blocks
#define CUDA_GET_THREADS(N, BLOCKS) ((N + BLOCKS - 1) / BLOCKS)
// CUDA: grid stride loop
#define CUDA_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// CUDA: various check for different function calls
#ifndef CUDA_CHECK
#define CUDA_CHECK(condition) \
    do { \
      cudaError_t error = (condition); \
      XDL_CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(condition) \
    do { \
      cublasStatus_t status = (condition); \
      XDL_CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " << xdl::common::cublasGetErrorString(status); \
    } while (0)
#endif

#ifndef CURAND_CHECK
#define CURAND_CHECK(condition) \
    do { \
      curandStatus_t status = condition; \
      XDL_CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " << xdl::common::curandGetErrorString(status); \
    } while (0)
#endif

#ifdef __CUDACC__
#define CUDA_CALL __device__
#define CUDA_XCALL __device__ __host__
#define CUDA_XCALL_INL __inline__ __device__ __host__
#else
#define CUDA_CALL
#define CUDA_XCALL
#define CUDA_XCALL_INL __inline__
#endif

#ifndef FLOAT_VALID
#define FLOAT_VALID(x)  (isnan(x) == false && isinf(x) == false)
#endif

/// GPU Assert
#ifndef CUDA_ASSERT
#define CUDA_ASSERT(FLAG, FUNC) \
    if (!(FLAG)) { \
      FUNC; \
      assert(0); \
    }
#endif

namespace xdl {
namespace common {
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);
bool HasGPU();
int GPUCount();
/// The mutex used to synchronize CUDA and NCCL operations
std::mutex& CudaMutex();
}  // namespace common
}  // namespace xdl

#endif // USE_GPU

#endif  // XDL_CORE_LIB_COMMON_DEFINES_H_

/*!
 * \file cuda_helpers.h
 * \brief Some cuda helper definitions.
 */
#pragma once

#include <memory>

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace blaze {

#define alignN(n, a) ((((n) + a - 1) / a) * a)

#define SWITCH_BLOCKDIM_NOTYPE(x, kernel, blocks, threads, size, stream, ...) \
  switch (x) {                                                          \
  case 512: kernel<512><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 256: kernel<256><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 128: kernel<128><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 64:  kernel< 64><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 32:  kernel< 32><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 16:  kernel< 16><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 8:   kernel<  8><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 4:   kernel<  4><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 2:   kernel<  2><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 1:   kernel<  1><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  }

#define SWITCH_BLOCKDIM(x, type, kernel, blocks, threads, size, stream, ...) \
  switch (x) {                                                          \
  case 512: kernel<type, 512><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 256: kernel<type, 256><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 128: kernel<type, 128><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 64:  kernel<type,  64><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 32:  kernel<type,  32><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 16:  kernel<type,  16><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 8:   kernel<type,   8><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 4:   kernel<type,   4><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 2:   kernel<type,   2><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  case 1:   kernel<type,   1><<<blocks, threads, size, stream>>>(__VA_ARGS__);break; \
  }

// Get the number of threads according to the size of the data. When upper
// equals true, the number of threads is the maximum upper bound, and vice versa
inline int GetThreadsNum(int data_size, bool upper = false) {
  if (upper) {
    // Return threadnum >= data_size if data_size <= 512
    if (data_size < 16) return 16;
    else if (data_size < 32) return 32;
    else if (data_size < 64) return 64;
    else if (data_size < 128) return 128;
    else if (data_size < 256) return 256;
    else return 512;
  } else {
    // Rerurn theadnum <= data_size
    if (data_size >= 512) return 512;
    else if (data_size >= 256) return 256;
    else if (data_size >= 128) return 128;
    else if (data_size >= 64) return 64;
    else if (data_size >= 32) return 32;
    else if (data_size >= 16) return 16;
    else return data_size;
  }
}

// Get the number of blocks, elem_per_thread represents the number of
// elements processed by each thread
inline int GetBlockNum(int block_size, int elem_per_thread = 16) {
  int z = block_size / elem_per_thread;
  if (z <= 0) return 1;
  else return z;
}

#ifdef USE_CUDA
// Get GPU device related properties
cudaDeviceProp* GetDeviceProp(int device_id);

// float16 declare
struct float16;

// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//
// template <class T>
// __global__ void foo(T* g_idata, T* g_odata) {
//   // Shared mem size is determined by the host app at run time
//   extern __shared__  T sdata[];
//   ...
//   doStuff(sdata);
//   ...
// }
//
// With this
//
// template<class T>
// __global__ void foo(T* g_idata, T* g_odata) {
//   // Shared mem size is determined by the host app at run time
//   SharedMemory<T> smem;
//   T* sdata = smem.GetPointer();
//   ...
//   doStuff(sdata);
//   ...
// }
//
template <typename T>
struct SharedMemory {
  __device__ T* GetPointer() {
    extern __device__ void error(void);
    error();
    return nullptr;
  }
};

template <>
struct SharedMemory<float16> {
  __device__ float16* GetPointer() {
    extern __shared__ float16 s_float16[];
    return s_float16;
  }
};

template <>
struct SharedMemory<float> {
  __device__ float* GetPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <>
struct SharedMemory<double> {
  __device__ double* GetPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

// CUDA >= 9.0, should call sync warp.
#if CUDA_VERSION < 9000
#define __syncwarp()
#endif

#endif // USE_CUDA

}  // namespace blaze

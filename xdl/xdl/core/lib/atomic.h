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
#ifndef XDL_CORE_LIB_ATOMIC_H_
#define XDL_CORE_LIB_ATOMIC_H_

#include <atomic>

namespace xdl {
namespace common {

#ifdef __CUDACC__
template <typename DType>
__inline__ __device__ DType gpu_atomic_add(const DType val, DType* address) {
  /// NOLINT
}

template <>
__inline__ __device__
float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <>
__inline__ __device__
unsigned int gpu_atomic_add(const unsigned int val, unsigned int* address) {
  return atomicAdd(address, val);
}

template <>
__inline__ __device__
int32_t gpu_atomic_add(const int32_t val, int32_t* address) {
  return atomicAdd(reinterpret_cast<int*>(address),
                   static_cast<int>(val));
}

/// double atomicAdd implementation taken from:
/// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
__inline__ __device__
double gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);  // NOLINT(runtime/int)
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  if (val == 0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif /// __CUDACC__

#ifndef _STDATOMIC_H
#define MEM_ORDER __ATOMIC_RELAXED
#define atomic_load_explicit(PTR, MO)                                \
    __extension__                                                    \
    ({                                                               \
       auto __atomic_load_ptr = (PTR);                               \
       __typeof__ (*__atomic_load_ptr) __atomic_load_tmp;            \
       __atomic_load (__atomic_load_ptr, &__atomic_load_tmp, (MO));  \
       __atomic_load_tmp;                                            \
     })

#define atomic_load(PTR)  atomic_load_explicit (PTR, MEM_ORDER)

#define atomic_compare_exchange_weak_explicit(PTR, VAL, DES, SUC, FAIL)             \
    __extension__                                                                   \
    ({                                                                              \
       auto __atomic_compare_exchange_ptr = (PTR);                                  \
       __typeof__ (*__atomic_compare_exchange_ptr) __atomic_compare_exchange_tmp    \
           = (DES);                                                                 \
       __atomic_compare_exchange (__atomic_compare_exchange_ptr, (VAL),             \
                                  &__atomic_compare_exchange_tmp, 1,                \
                                  (SUC), (FAIL));                                   \
     })

#define atomic_compare_exchange_weak(PTR, VAL, DES)                                 \
    atomic_compare_exchange_weak_explicit (PTR, VAL, DES, MEM_ORDER,                \
                                           MEM_ORDER)
#endif  // _STDATOMIC_H

template <typename Dtype>
__inline__ Dtype cpu_atomic_add(const Dtype val, Dtype* address) {
  Dtype current = atomic_load(address);
  while (!atomic_compare_exchange_weak(address, &current, current + val));
  return current;
}

}  // namespace common
}  // namespace xdl

#endif  // XDL_CORE_LIB_ATOMIC_H_

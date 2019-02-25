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

#ifndef TDM_SERVING_UTIL_CONCURRENCY_BARRIER_H_
#define TDM_SERVING_UTIL_CONCURRENCY_BARRIER_H_

namespace tdm_serving {
namespace util {

#ifdef __GNUC__
inline void CompilerBarrier() { __asm__ __volatile__("": : :"memory"); }
# if defined __i386__ || defined __x86_64__
inline void MemoryBarrier() { __asm__ __volatile__("mfence": : :"memory"); }
inline void MemoryReadBarrier() { __asm__ __volatile__("lfence"::: "memory"); }
inline void MemoryWriteBarrier() { __asm__ __volatile__("sfence"::: "memory"); }
# else
inline void MemoryBarrier() { CompilerBarrier(); }
inline void MemoryReadBarrier() { MemoryBarrier(); }
inline void MemoryWriteBarrier() { MemoryBarrier(); }
# endif
#else
inline void CompilerBarrier() { volatile int n = 0; n = 0; }
inline void MemoryBarrier() { CompilerBarrier(); }
inline void MemoryReadBarrier() { MemoryBarrier(); }
inline void MemoryWriteBarrier() { MemoryBarrier(); }
#endif

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_CONCURRENCY_BARRIER_H_


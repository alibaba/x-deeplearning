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
#ifndef XDL_CORE_LIB_BINARY_SEARCH_H_
#define XDL_CORE_LIB_BINARY_SEARCH_H_

namespace xdl {

template <typename ForwardIt, typename T>
__host__ __device__ ForwardIt LowerBound(ForwardIt first,
                                         ForwardIt last,
                                         const T& value) {
  ForwardIt it;
  ptrdiff_t count = last - first, step;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (*it < value) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

template <typename T>
__host__ __device__ int BinarySearch(const T* src, size_t sz, const T& v) {
  int l = 0, r = sz - 1, m;
  while (l <= r) {
    m = l + (r - l) / 2;
    if (v == src[m]) {
      return m;
    } else if (v < src[m]) {
      r = m - 1;
    } else {
      l = m + 1;
    }
  }
  return -1;
}

template <typename T>
__host__ __device__ int BinarySearch2(const T* src, size_t sz,
                                      const T& v1, const T& v2) {
  int l = 0, r = sz - 1, m;
  while (l <= r) {
    m = l + (r - l) / 2;
    if (v1 == src[2*m] && v2 == src[2*m+1]) {
      return m;
    } else if (v1 < src[2*m] || (v1 == src[2*m] && v2 < src[2*m+1])) {
      r = m - 1;
    } else {
      l = m + 1;
    }
  }
  return -1;
}

}  // namespace xdl

#endif  // XDL_CORE_LIB_BINARY_SEARCH_H_

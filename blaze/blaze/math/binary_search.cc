/*
 * \file binary_search.cc
 * \desc The impl of binary search
 */
#include "blaze/math/binary_search.h"

#include <math.h>

namespace blaze {

template<>
void BinarySearch<int32_t>(const int32_t *data,
                           int N,
                           const int32_t *keys,
                           int M,
                           int32_t *result) {
  const int vsize = 64;
  // calc biggest power of 2 smaller than N
  int32_t powerOfTwo = pow(2, floor(log2(N - 1)));
  // used to divide a problem into two problems
  int32_t splitIndex = N - powerOfTwo;
  // with sizees guaranteed to be powers of 2
  int32_t splitValue = data[splitIndex];

  int processed = 0;
  for (; processed + vsize < M; processed += vsize) {
    int32_t *result_vector = result + processed;
    const int32_t *keys_vector = keys + processed;
    __m128i xm_idxvec = _mm_set_epi32(splitIndex, splitIndex, splitIndex, splitIndex);
    __m128i xm_valvec = _mm_set_epi32(splitValue, splitValue, splitValue, splitValue);
    // Prepare the first index in the search
    for (auto i = 0; i < vsize; i += 4) {
      __m128i xm_vals = _mm_load_si128((__m128i*)(keys_vector + i));
      xm_vals = _mm_andnot_si128(_mm_cmplt_epi32(xm_vals, xm_valvec), xm_idxvec);
      _mm_store_si128((__m128i*)(result_vector + i), xm_vals);
    }
    // Then, for each search phase, perform it for all the elements in a vector
    for (auto j = powerOfTwo >> 1; j >= 1; j = j >> 1) {
      const int32_t *data_shifted = data + j;
      __m128i xm_jvec = _mm_set_epi32(j, j, j, j);
      for (auto i = 0; i < vsize; i += 4) {
        __m128i xm_idxvec = _mm_load_si128((__m128i*)(result_vector + i));
        int32_t cmpval0 = data_shifted[result_vector[i + 0]];
        int32_t cmpval1 = data_shifted[result_vector[i + 1]];
        int32_t cmpval2 = data_shifted[result_vector[i + 2]];
        int32_t cmpval3 = data_shifted[result_vector[i + 3]];
        __m128i xm_cmpvalvec = _mm_set_epi32(cmpval3, cmpval2, cmpval1, cmpval0);
        __m128i xm_valvec = _mm_load_si128((__m128i*)(keys_vector + i));
        xm_idxvec = _mm_add_epi32(xm_idxvec, _mm_andnot_si128(_mm_cmplt_epi32(xm_valvec, xm_cmpvalvec), xm_jvec));
        _mm_store_si128((__m128i*)(result_vector + i), xm_idxvec);
      }
    }

    // Make missed result -1
    for (auto i = 0; i < vsize; i += 4) {
      const int32_t cmpval0 = data[result_vector[i + 0]];
      const int32_t cmpval1 = data[result_vector[i + 1]];
      const int32_t cmpval2 = data[result_vector[i + 2]];
      const int32_t cmpval3 = data[result_vector[i + 3]];
      __m128i xm_cmpvalvec = _mm_set_epi32(cmpval3, cmpval2, cmpval1, cmpval0);
      xm_valvec = _mm_load_si128((__m128i *) (keys_vector + i));

      __m128i xm_missvec = _mm_set_epi32(-1, -1, -1, -1);
      xm_idxvec = _mm_load_si128((__m128i *) (result_vector + i));
      xm_idxvec = _mm_and_si128(_mm_cmpeq_epi32(xm_cmpvalvec, xm_valvec), xm_idxvec);
      xm_idxvec = _mm_add_epi32(xm_idxvec,
                                _mm_andnot_si128(_mm_cmpeq_epi32(xm_cmpvalvec, xm_valvec), xm_missvec));
      _mm_store_si128((__m128i *) (result_vector + i), xm_idxvec);
    }
  }

  for (auto i = processed; i < M; i++) {
    auto key = keys[i];
    auto p = 0;
    if (key >= splitValue) {
      p = splitIndex;
    } else {
      p = 0;
    }

    if (key == data[p]) {
      result[i] = p;
    } else {
      result[i] = -1;
      for (auto j = powerOfTwo >> 1; j >= 1; j = j >> 1) {
        if (key == data[p + j]) {
          result[i] = p + j;
          break;
        }
        if (key > data[p + j]) {
          p += j;
        }
      }
    }
  }
}

template<>
void BinarySearch<int64_t>(const int64_t *data,
                           int N,
                           const int64_t *keys,
                           int M,
                           int64_t *result) {
  const int vsize = 64;
  // calc biggest power of 2 smaller than N
  int64_t powerOfTwo = pow(2, floor(log2(N - 1)));
  // used to divide a problem into two problems
  int64_t splitIndex = N - powerOfTwo;
  // with sizees guaranteed to be powers of 2
  int64_t splitValue = data[splitIndex];

  int processed = 0;
  for (; processed + vsize < M; processed += vsize) {
    int64_t *result_vector = result + processed;
    const int64_t *keys_vector = keys + processed;
    __m256i xm_idxvec = _mm256_set_epi64x(splitIndex, splitIndex, splitIndex, splitIndex);
    __m256i xm_valvec = _mm256_set_epi64x(splitValue, splitValue, splitValue, splitValue);
    // Prepare the first index in the search
    for (auto i = 0; i < vsize; i += 4) {
      __m256i xm_vals = _mm256_load_si256((__m256i*)(keys_vector + i));
      xm_vals = _mm256_andnot_si256(_mm256_cmpgt_epi64(xm_valvec, xm_vals), xm_idxvec);
      _mm256_store_si256((__m256i*)(result_vector + i), xm_vals);
    }
    // Then, for each search phase, perform it for all the elements in a vector
    for (auto j = powerOfTwo >> 1; j >= 1; j = j >> 1) {
      const int64_t *data_shifted = data + j;
      __m256i xm_jvec = _mm256_set_epi64x(j, j, j, j);
      for (auto i = 0; i < vsize; i += 4) {
        __m256i xm_idxvec = _mm256_load_si256((__m256i*)(result_vector + i));
        int64_t cmpval0 = data_shifted[result_vector[i + 0]];
        int64_t cmpval1 = data_shifted[result_vector[i + 1]];
        int64_t cmpval2 = data_shifted[result_vector[i + 2]];
        int64_t cmpval3 = data_shifted[result_vector[i + 3]];
        __m256i xm_cmpvalvec = _mm256_set_epi64x(cmpval3, cmpval2, cmpval1, cmpval0);
        __m256i xm_valvec = _mm256_load_si256((__m256i*)(keys_vector + i));
        xm_idxvec = _mm256_add_epi64(xm_idxvec, _mm256_andnot_si256(_mm256_cmpgt_epi64(xm_cmpvalvec, xm_valvec), xm_jvec));
        _mm256_store_si256((__m256i*)(result_vector + i), xm_idxvec);
      }
    }

    // Make missed result -1
    for (auto i = 0; i < vsize; i += 4) {
      const int64_t cmpval0 = data[result_vector[i + 0]];
      const int64_t cmpval1 = data[result_vector[i + 1]];
      const int64_t cmpval2 = data[result_vector[i + 2]];
      const int64_t cmpval3 = data[result_vector[i + 3]];
      __m256i xm_cmpvalvec = _mm256_set_epi64x(cmpval3, cmpval2, cmpval1, cmpval0);
      xm_valvec = _mm256_load_si256((__m256i *) (keys_vector + i));

      __m256i xm_missvec = _mm256_set_epi64x(-1, -1, -1, -1);
      xm_idxvec = _mm256_load_si256((__m256i *) (result_vector + i));
      xm_idxvec = _mm256_and_si256(_mm256_cmpeq_epi64(xm_cmpvalvec, xm_valvec), xm_idxvec);
      xm_idxvec = _mm256_add_epi64(xm_idxvec,
                                   _mm256_andnot_si256(_mm256_cmpeq_epi64(xm_cmpvalvec, xm_valvec), xm_missvec));
      _mm256_store_si256((__m256i *) (result_vector + i), xm_idxvec);
    }
  }

  for (auto i = processed; i < M; i++) {
    auto key = keys[i];
    auto p = 0;
    if (key >= splitValue) {
      p = splitIndex;
    } else {
      p = 0;
    }

    if (key == data[p]) {
      result[i] = p;
    } else {
      result[i] = -1;
      for (auto j = powerOfTwo >> 1; j >= 1; j = j >> 1) {
        if (key == data[p + j]) {
          result[i] = p + j;
          break;
        }
        if (key > data[p + j]) {
          p += j;
        }
      }
    }
  }
}

}  // namespace blaze
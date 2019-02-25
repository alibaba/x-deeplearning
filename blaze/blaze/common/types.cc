/*
 * \file types.cc
 * \desc The types in blaze.
 */
#include "blaze/common/types.h"

namespace blaze {

void float2half(const float* floats, size_t size, float16* halfs) {
  size_t k = 0, vsize = size - size % 8;
  for (; k < vsize; k += 8) {
    __m256 float_vector = _mm256_loadu_ps(floats + k);
    __m128i half_vector = _mm256_cvtps_ph(float_vector, 0);
    _mm_storeu_si128((__m128i*)(halfs + k), half_vector);
  }
  for (; k < size; ++k) {
    halfs[k].x = _cvtss_sh(floats[k], 0);
  }
}

void half2float(const float16* halfs, size_t size, float* floats) {
  size_t k = 0, vsize = size - size % 8;
  for (; k < vsize; k += 8) {
    __m256 vector = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(halfs + k)));
    _mm256_storeu_ps(floats, vector);
  }
  for (; k < size; ++k) {
    floats[k] = _cvtsh_ss(halfs[k].x);
  }
}

}  // namespace blaze

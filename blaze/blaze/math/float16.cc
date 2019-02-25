/*
 * \file float16.cc
 * \desc The float16 utility 
 */
#include "blaze/math/float16.h"

namespace blaze {

static inline void float2half256(const float* s, float16* h) {
  __m256 float_vector = _mm256_loadu_ps(s);
  __m128i half_vector = _mm256_cvtps_ph(float_vector, 0);
  _mm_storeu_si128((__m128i*)(h), half_vector);
}

static inline void half2float256(const float16* h, float* s) {
  __m256 vector = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(h)));
  _mm256_storeu_ps(s, vector);
}

template <>
void VML_Add<float16, CPUContext>(const int N, const float16* a, const float16* b, float16* y, CPUContext*) {
  float a_f[8];
  float b_f[8];
  float y_f[8];
  int k = 0, vsize = N - N % 8;
  for (; k < vsize; k += 8) {
    half2float256(a + k, a_f);
    half2float256(b + k, b_f);
    VML_Add<float, CPUContext>(8, a_f, b_f, y_f, nullptr);
    float2half256(y_f, y + k);
  }
  for (; k < N; ++k) {
    y[k] = a[k] + b[k];
  }
}

template <>
void VML_Sub<float16, CPUContext>(const int N, const float16* a, const float16* b, float16* y, CPUContext*) {
  float a_f[8];
  float b_f[8];
  float y_f[8];
  int k = 0, vsize = N - N % 8;
  for (; k < vsize; k += 8) {
    half2float256(a + k, a_f);
    half2float256(b + k, b_f);
    VML_Sub<float, CPUContext>(8, a_f, b_f, y_f, nullptr);
    float2half256(y_f, y + k);
  }
  for (; k < N; ++k) {
    y[k] = a[k] - b[k];
  }
}

template <>
void VML_Mul<float16, CPUContext>(const int N, const float16* a, const float16* b, float16* y, CPUContext*) {
  float a_f[8];
  float b_f[8];
  float y_f[8];
  int k = 0, vsize = N - N % 8;
  for (; k < vsize; k += 8) {
    half2float256(a + k, a_f);
    half2float256(b + k, b_f);
    VML_Mul<float, CPUContext>(8, a_f, b_f, y_f, nullptr);
    float2half256(y_f, y + k);
  }
  for (; k < N; ++k) {
    y[k] = a[k] * b[k];
  }
}

template <>
void VML_Div<float16, CPUContext>(const int N, const float16* a, const float16* b, float16* y, CPUContext*) {
  float a_f[8];
  float b_f[8];
  float y_f[8];
  int k = 0, vsize = N - N % 8;
  for (; k < vsize; k += 8) {
    half2float256(a + k, a_f);
    half2float256(b + k, b_f);
    VML_Div<float, CPUContext>(8, a_f, b_f, y_f, nullptr);
    float2half256(y_f, y + k);
  }
  for (; k < N; ++k) {
    y[k] = a[k] / b[k];
  }
}

}  // namespace blaze

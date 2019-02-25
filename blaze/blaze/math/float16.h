/*
 * \file float16.h
 * \desc The float16 utility 
 */
#pragma once

#include "blaze/common/types.h"

#include "blaze/math/vml.h"

namespace blaze {

#ifdef __CUDACC__
BLAZE_INLINE BLAZE_DEVICE float16 operator+(float16 a, float16 b) {
  __half ret = __hadd(*(__half*)&a, *(__half*)&b); 
  return *(float16*)&ret;
}

BLAZE_INLINE BLAZE_DEVICE float16 operator-(float16 a, float16 b) {
  __half ret = __hsub(*(__half*)&a, *(__half*)&b); 
  return *(float16*)&ret;
}

BLAZE_INLINE BLAZE_DEVICE float16 operator*(float16 a, float16 b) {
  __half ret = __hmul(*(__half*)&a, *(__half*)&b); 
  return *(float16*)&ret;
}

BLAZE_INLINE BLAZE_DEVICE float16 operator/(float16 a, float16 b) {
#if CUDA_VERSION >= 9000
  __half ret = __hdiv(*(__half*)&a, *(__half*)&b); 
#else
  __half ret = hdiv(*(__half*)&a, *(__half*)&b); 
#endif
  return *(float16*)&ret;
}
#else
BLAZE_INLINE float16 operator+(float16 a, float16 b) {
  float a_f, b_f;
  half2float(&a, 1, &a_f);
  half2float(&b, 1, &b_f);
  float ret = a_f + b_f;
  float16 ret_h;
  float2half(&ret, 1, &ret_h);
  return ret_h;
}

BLAZE_INLINE float16 operator-(float16 a, float16 b) {
  float a_f, b_f;
  half2float(&a, 1, &a_f);
  half2float(&b, 1, &b_f);
  float ret = a_f - b_f;
  float16 ret_h;
  float2half(&ret, 1, &ret_h);
  return ret_h;
}

BLAZE_INLINE float16 operator*(float16 a, float16 b) {
  float a_f, b_f;
  half2float(&a, 1, &a_f);
  half2float(&b, 1, &b_f);
  float ret = a_f * b_f;
  float16 ret_h;
  float2half(&ret, 1, &ret_h);
  return ret_h;
}

BLAZE_INLINE float16 operator/(float16 a, float16 b) {
  float a_f, b_f;
  half2float(&a, 1, &a_f);
  half2float(&b, 1, &b_f);
  float ret = a_f / b_f;
  float16 ret_h;
  float2half(&ret, 1, &ret_h);
  return ret_h;
}
#endif

}  // namespace blaze


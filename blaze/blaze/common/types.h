/*
 * \file types.h
 * \desc The types in blaze.
 */
#pragma once

#include <x86intrin.h>
#include <immintrin.h>
#include <f16cintrin.h>

#include <string>

#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

inline bool IsIntegerType(int data_type) {
  switch (data_type) {
    case DataType::kInt32:
    case DataType::kUInt8:
    case DataType::kInt8:
    case DataType::kUInt16:
    case DataType::kInt16:
    case DataType::kInt64:
      return true;
    default:
      return false;
  }
}

inline bool IsFloatType(int data_type) {
  switch (data_type) {
    case kFloat:
    case kDouble:
    case kFloat16:
      return true;
    default:
      return false;
  }
}

struct float16; 

// float16 <-> float32 convertion
void float2half(const float* floats, size_t size, float16* halfs);
void half2float(const float16* halfs, size_t size, float* floats);

struct ALIGNED(2) float16 {
  uint16_t x;
  
  BLAZE_INLINE float16() { }
#ifdef __CUDACC__
  BLAZE_INLINE BLAZE_DEVICE float16(const float value) {
    __half val = __float2half(value);
#if CUDA_VERSION >= 9000
    this->x = ((__half_raw)val).x;
#else
    this->x = val.x;
#endif
  }
  BLAZE_INLINE BLAZE_DEVICE operator float() const {
    __half val;
#if CUDA_VERSION >= 9000
    __half_raw raw;
    raw.x = x;
    val = raw;
#else
    val.x = x;
#endif
    return __half2float(val);
  }
  BLAZE_INLINE BLAZE_DEVICE float16 operator+=(const float16& other) {
    return *this = (*this + other);
  }
#else
  BLAZE_INLINE float16(const float value) {
    float2half(&value, 1, this);
  }
  BLAZE_INLINE operator float() const { float a; half2float(this, 1, &a); return a; }
#endif
};

template <typename DType>
struct TypeFlag;

template <>
struct TypeFlag<float> {
  static const int kFlag = DataType::kFloat;
};

template <>
struct TypeFlag<double> {
  static const int kFlag = DataType::kDouble;
};

template <>
struct TypeFlag<float16> {
  static const int kFlag = DataType::kFloat16;
};

template <>
struct TypeFlag<int32_t> {
  static const int kFlag = DataType::kInt32;
};

template <>
struct TypeFlag<bool> {
  static const int kFlag = DataType::kBool;
};

template <>
struct TypeFlag<uint8_t> {
  static const int kFlag = DataType::kUInt8;
};

template <>
struct TypeFlag<int8_t> {
  static const int kFlag = DataType::kInt8;
};

template <>
struct TypeFlag<uint16_t> {
  static const int kFlag = DataType::kUInt16;
};

template <>
struct TypeFlag<int16_t> {
  static const int kFlag = DataType::kInt16;
};

template <>
struct TypeFlag<int64_t> {
  static const int kFlag = DataType::kInt64;
};

inline size_t DataTypeSize(int data_type) {
  switch (data_type) {
    case DataType::kFloat:
      return sizeof(float);
    case DataType::kDouble:
      return sizeof(double);
    case DataType::kFloat16:
      return sizeof(float16);
    case DataType::kInt32:
      return sizeof(int32_t);
    case DataType::kBool:
      return sizeof(bool);
    case DataType::kUInt8:
      return sizeof(uint8_t);
    case DataType::kInt8:
      return sizeof(int8_t);
    case DataType::kUInt16:
      return sizeof(uint16_t);
    case DataType::kInt16:
      return sizeof(int16_t);
    case DataType::kInt64:
      return sizeof(int64_t);
    default:
      BLAZE_THROW("Not supported type=", data_type);
  }
}

// TYPE_SWITCH which is VALUE_TYPE_SWITCH
// The cpu platform data type switch.
#ifndef TYPE_SWITCH
#define TYPE_SWITCH(type, DType, ...)              \
   switch (type) {                                 \
     case kFloat:                                  \
      {                                            \
        typedef float DType;                       \
        {__VA_ARGS__}                              \
      }                                            \
      break;                                       \
     default:                                      \
      {                                            \
        BLAZE_THROW("Unsupported type:", type);    \
      }                                            \
    }
#endif

// The cuda platform data type switch.
#ifndef TYPE_SWITCH_ON_CUDA
#define TYPE_SWITCH_ON_CUDA(type, DType, ...)            \
   switch (type) {                                       \
     case kFloat:                                        \
      {                                                  \
        typedef float DType;                             \
        {__VA_ARGS__}                                    \
      }                                                  \
      break;                                             \
     case kFloat16:                                      \
      {                                                  \
        typedef float16 DType;                           \
        {__VA_ARGS__}                                    \
      }                                                  \
      break;                                             \
     default:                                            \
      {                                                  \
        BLAZE_THROW("Unsupported type: ", type);         \
      }                                                  \
   }
#endif

// Each platform has it's own supported data type, for example
// On CPU platform, only float data type is supported; On GPU
// platform, the half and float data type are supported.
#ifndef TYPE_SWITCH_WITH_CTX
#define TYPE_SWITCH_WITH_CTX(ctx, type, DType, ...)        \
   if (ctx.device_type() == kCPU) {                        \
     TYPE_SWITCH(type, DType, __VA_ARGS__)                 \
   } else if (ctx.device_type() == kCUDA) {                \
     TYPE_SWITCH_ON_CUDA(type, DType, __VA_ARGS__)         \
   } else {                                                \
     BLAZE_THROW("Unsupported ctx: ", ctx.device_type());  \
   }
#endif

// Convert the data type in Model into real pass operation data type 
// In kernel fusion phase, the data movement and precompute will use
// kFloat instead of low-precision.(As on CPU platform, low-precision is not
// supported)
inline DataType DataType2PassDataType(int data_type) {
  switch (data_type) {
    case kFloat16:
    case kFloat:
      return kFloat;
    default:
      BLAZE_THROW("Unsupported data type: ", data_type);
  }
}

// The ID type switch, The supported id data types are:
// int32 and int64
#ifndef ID_TYPE_SWITCH
#define ID_TYPE_SWITCH(type, DType, ...)           \
   switch (type) {                                 \
     case kInt32:                                  \
      {                                            \
        typedef int32_t DType;                     \
        {__VA_ARGS__}                              \
      }                                            \
      break;                                       \
     case kInt64:                                  \
      {                                            \
        typedef int64_t DType;                     \
        {__VA_ARGS__}                              \
      }                                            \
      break;                                       \
     default:                                      \
      {                                            \
        BLAZE_THROW("Unsupported type", type);     \
      }                                            \
    }
#endif

// The conctant fill type switch, which is used for model
// weight loading.
#ifndef CONSTANT_FILL_TYPE_SWITCH
#define CONSTANT_FILL_TYPE_SWITCH(type, DType, ...)    \
   switch (type) {                                     \
     case kFloat:                                      \
      {                                                \
        typedef float DType;                           \
        {__VA_ARGS__}                                  \
      }                                                \
      break;                                           \
     case kFloat16:                                    \
      {                                                \
        typedef float16 DType;                         \
        {__VA_ARGS__}                                  \
      }                                                \
      break;                                           \
     case kInt32:                                      \
      {                                                \
        typedef int32_t DType;                         \
        {__VA_ARGS__}                                  \
      }                                                \
      break;                                           \
     case kInt64:                                      \
      {                                                \
        typedef int64_t DType;                         \
        {__VA_ARGS__}                                  \
      }                                                \
      break;                                           \
     default:                                          \
      {                                                \
        BLAZE_THROW("Unsupported type", type);         \
      }                                                \
    }
#endif

}  // namespace blaze

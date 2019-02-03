/*!
 * \file defines.h
 * \desc store defines
 */
#pragma once

namespace blaze {
namespace store {

// EMBEDDING_KEY_TYPE_SWITCH which is store key type switch.
#ifndef EMBEDDING_KEY_TYPE_SWITCH
#define EMBEDDING_KEY_TYPE_SWITCH(type, K_DType, ...)            \
   switch (type) {                                               \
     case kInt64:                                                \
      {                                                          \
        typedef int64_t K_DType;                                 \
        {__VA_ARGS__}                                            \
      }                                                          \
      break;                                                     \
     default:                                                    \
      {                                                          \
        BLAZE_THROW("Unsupported key type:", type);              \
      }                                                          \
    }
#endif

// EMBEDDING_VALUE_TYPE_SWITCH which is store value type switch.
#ifndef EMBEDDING_VALUE_TYPE_SWITCH
#define EMBEDDING_VALUE_TYPE_SWITCH(type, V_DType, ...)          \
   switch (type) {                                               \
     case kFloat:                                                \
      {                                                          \
        typedef float V_DType;                                   \
        {__VA_ARGS__}                                            \
      }                                                          \
      break;                                                     \
     case kFloat16:                                              \
      {                                                          \
        typedef float16 V_DType;                                 \
        {__VA_ARGS__}                                            \
      }                                                          \
      break;                                                     \
     default:                                                    \
      {                                                          \
        BLAZE_THROW("Unsupported value type:", type);            \
      }                                                          \
    }
#endif

// EMBEDDING_KEY_TYPE_SWITCH which is store id num type switch.
#ifndef EMBEDDING_NUM_TYPE_SWITCH
#define EMBEDDING_NUM_TYPE_SWITCH(type, N_DType, ...)            \
   switch (type) {                                               \
     case kInt32:                                                \
      {                                                          \
        typedef int32_t N_DType;                                 \
        {__VA_ARGS__}                                            \
      }                                                          \
      break;                                                     \
     default:                                                    \
      {                                                          \
        BLAZE_THROW("Unsupported num type:", type);              \
      }                                                          \
    }
#endif

}  // namespace store
}  // namespace blaze

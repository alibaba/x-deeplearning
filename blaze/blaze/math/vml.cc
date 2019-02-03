/*
 * \file vml.cc
 * \brief The VML routine on CPU Architecture
 */
#include "blaze/math/vml.h"

#include "blaze/math/activation.h"
#include "blaze/math/float16.h"
#include <omp.h>

#ifdef USE_MKL
#endif

namespace blaze {

#ifdef USE_MKL

#ifndef DECLARE_VML_FUNCTION_IMPL
#define DECLARE_VML_FUNCTION_IMPL(T, FuncName, OriginalFunc, ...)                 \
  template <>                                                                     \
  void FuncName<T, CPUContext>(const int N, const T* x, T* y, CPUContext*) {      \
    OriginalFunc(N, x, y, ##__VA_ARGS__);                                         \
  }
#endif

DECLARE_VML_FUNCTION_IMPL(float, VML_Exp, vmsExp, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE)
DECLARE_VML_FUNCTION_IMPL(double, VML_Exp, vmdExp, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE)

DECLARE_VML_FUNCTION_IMPL(float, VML_Log, vsLn)
DECLARE_VML_FUNCTION_IMPL(double, VML_Log, vdLn)

DECLARE_VML_FUNCTION_IMPL(float, VML_Cos, vsCos)
DECLARE_VML_FUNCTION_IMPL(double, VML_Cos, vdCos)

DECLARE_VML_FUNCTION_IMPL(float, VML_Acos, vsAcos)
DECLARE_VML_FUNCTION_IMPL(double, VML_Acos, vdAcos)

DECLARE_VML_FUNCTION_IMPL(float, VML_Sin, vsSin)
DECLARE_VML_FUNCTION_IMPL(double, VML_Sin, vdSin)

DECLARE_VML_FUNCTION_IMPL(float, VML_Asin, vsAsin)
DECLARE_VML_FUNCTION_IMPL(double, VML_Asin, vdAsin)

DECLARE_VML_FUNCTION_IMPL(float, VML_Tan, vsTan)
DECLARE_VML_FUNCTION_IMPL(double, VML_Tan, vdTan)

DECLARE_VML_FUNCTION_IMPL(float, VML_Tanh, vsTanh)
DECLARE_VML_FUNCTION_IMPL(double, VML_Tanh, vdTanh)

DECLARE_VML_FUNCTION_IMPL(float, VML_Atan, vsAtan)
DECLARE_VML_FUNCTION_IMPL(double, VML_Atan, vdAtan)

DECLARE_VML_FUNCTION_IMPL(float, VML_Abs, vsAbs)
DECLARE_VML_FUNCTION_IMPL(double, VML_Abs, vdAbs)

DECLARE_VML_FUNCTION_IMPL(float, VML_Sqrt, vsSqrt)
DECLARE_VML_FUNCTION_IMPL(double, VML_Sqrt, vdSqrt)

#undef DECLARE_VML_FUNCTION_IMPL

#ifndef DECLARE_VML_POWX_FUNCTION_IMPL
#define DECLARE_VML_POWX_FUNCTION_IMPL(T, OriginalFunc)                        \
  template <>                                                                  \
  void VML_Powx<T, CPUContext>(const int N, const T* a, T b, T* y, CPUContext*) {  \
    OriginalFunc(N, a, b, y);                                                  \
  }
#endif

DECLARE_VML_POWX_FUNCTION_IMPL(float, vsPowx)
DECLARE_VML_POWX_FUNCTION_IMPL(double, vdPowx)

#undef DECLARE_VML_POWX_FUNCTION_IMPL

#ifndef DECLARE_VML_BINARY_OP_FUNCTION_IMPL
#define DECLARE_VML_BINARY_OP_FUNCTION_IMPL(T, FuncName, OriginalFunc)              \
  template <>                                                                       \
  void FuncName<T, CPUContext>(const int N, const T* a, const T* b, T* y, CPUContext*) { \
    OriginalFunc(N, a, b, y);     \
  }
#endif

DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Add, vsAdd)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Add, vdAdd)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Sub, vsSub)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Sub, vdSub)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Mul, vsMul)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Mul, vdMul)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Div, vsDiv)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Div, vdDiv)

#undef DECLARE_VML_BINARY_OP_FUNCTION_IMPL

#else  // NOT USE_MPI

#ifndef DECLARE_VML_FUNCTION_IMPL
#define DECLARE_VML_FUNCTION_IMPL(T, FuncName, OriginalFunc)                 \
  template <>                                                                     \
  void FuncName<T, CPUContext>(const int N, const T* x, T* y, CPUContext*) {      \
    for (auto i = 0; i < N; ++i) { \
      y[i] = OriginalFunc(x[i]);                                         \
    } \
  }
#endif

DECLARE_VML_FUNCTION_IMPL(float, VML_Exp, std::exp)
DECLARE_VML_FUNCTION_IMPL(double, VML_Exp, std::exp)

DECLARE_VML_FUNCTION_IMPL(float, VML_Log, std::log)
DECLARE_VML_FUNCTION_IMPL(double, VML_Log, std::log)

DECLARE_VML_FUNCTION_IMPL(float, VML_Cos, std::cos)
DECLARE_VML_FUNCTION_IMPL(double, VML_Cos, std::cos)

DECLARE_VML_FUNCTION_IMPL(float, VML_Acos, std::acos)
DECLARE_VML_FUNCTION_IMPL(double, VML_Acos, std::acos)

DECLARE_VML_FUNCTION_IMPL(float, VML_Sin, std::sin)
DECLARE_VML_FUNCTION_IMPL(double, VML_Sin, std::sin)

DECLARE_VML_FUNCTION_IMPL(float, VML_Asin, std::asin)
DECLARE_VML_FUNCTION_IMPL(double, VML_Asin, std::asin)

DECLARE_VML_FUNCTION_IMPL(float, VML_Tan, std::tan)
DECLARE_VML_FUNCTION_IMPL(double, VML_Tan, std::tan)

DECLARE_VML_FUNCTION_IMPL(float, VML_Tanh, std::tanh)
DECLARE_VML_FUNCTION_IMPL(double, VML_Tanh, std::tanh)

DECLARE_VML_FUNCTION_IMPL(float, VML_Atan, std::atan)
DECLARE_VML_FUNCTION_IMPL(double, VML_Atan, std::atan)

DECLARE_VML_FUNCTION_IMPL(float, VML_Abs, std::abs)
DECLARE_VML_FUNCTION_IMPL(double, VML_Abs, std::abs)

DECLARE_VML_FUNCTION_IMPL(float, VML_Sqrt, std::sqrt)
DECLARE_VML_FUNCTION_IMPL(double, VML_Sqrt, std::sqrt)

#undef DECLARE_VML_FUNCTION_IMPL

#ifndef DECLARE_VML_POWX_FUNCTION_IMPL
#define DECLARE_VML_POWX_FUNCTION_IMPL(T, OriginalFunc)                        \
  template <>                                                                  \
  void VML_Powx<T, CPUContext>(const int N, const T* a, T b, T* y, CPUContext*) {  \
    for (auto i = 0; i < N; ++i)                                               { \
      y[i] = OriginalFunc(a[i], b);                                                  \
    } \
  }
#endif

DECLARE_VML_POWX_FUNCTION_IMPL(float, std::pow)
DECLARE_VML_POWX_FUNCTION_IMPL(double, std::pow)

#undef DECLARE_VML_POWX_FUNCTION_IMPL

#ifndef DECLARE_VML_BINARY_OP_FUNCTION_IMPL
#define DECLARE_VML_BINARY_OP_FUNCTION_IMPL(T, FuncName, Operand)                        \
  template <>                                                                            \
  void FuncName<T, CPUContext>(const int N, const T* a, const T* b, T* y, CPUContext*) { \
    for (auto i = 0; i < N; ++i) {                                                       \
      y[i] = a[i] Operand b[i];                                                          \
    }                                                                                    \
  }
#endif

DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Add, +)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Add, +)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Sub, -)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Sub, -)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Mul, *)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Mul, *)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(float, VML_Div, /)
DECLARE_VML_BINARY_OP_FUNCTION_IMPL(double, VML_Div, /)

#undef DECLARE_VML_BINARY_OP_FUNCTION_IMPL

#endif

#ifndef DECLARE_VML_SET_FUNCTION_IMPL
#define DECLARE_VML_SET_FUNCTION_IMPL(T)                                  \
  template <>                                                             \
  void VML_Set<T, CPUContext>(const int N, T* a, T v, CPUContext*) {      \
    for (int i = 0; i < N; ++i) {                                         \
      a[i] = v;                                                           \
    }                                                                     \
  }
#endif

DECLARE_VML_SET_FUNCTION_IMPL(float16)
DECLARE_VML_SET_FUNCTION_IMPL(float)
DECLARE_VML_SET_FUNCTION_IMPL(double)

#undef DECLARE_VML_SET_FUNCTION_IMPL

#ifndef DECLARE_VML_SET2_FUNCTION_IMPL
#define DECLARE_VML_SET2_FUNCTION_IMPL(DstT, SrcT)                                             \
  template <>                                                                                  \
  void VML_Set<DstT, SrcT, CPUContext>(const int N, DstT* dst, const SrcT* src, CPUContext*) { \
    for (int i = 0; i < N; ++i) {                                                              \
      dst[i] = src[i];                                                                         \
    }                                                                                          \
  }
#endif

DECLARE_VML_SET2_FUNCTION_IMPL(float, float16)
DECLARE_VML_SET2_FUNCTION_IMPL(float16, float)
DECLARE_VML_SET2_FUNCTION_IMPL(float16, float16)
DECLARE_VML_SET2_FUNCTION_IMPL(float, float)

#undef DECLARE_VML_SET2_FUNCTION_IMPL

#ifndef DECLARE_VML_WHERE_FUNCTION_IMPL
#define DECLARE_VML_WHERE_FUNCTION_IMPL(T1, T2)                                       \
  template <>                                                                         \
  void VML_Where<T1, T2, CPUContext>(const int N, const T1* condition,                \
                                     const T2* a, const T2* b, T2* y, CPUContext*) {  \
    for (int i = 0; i < N; ++i) {                                                     \
      y[i] = condition[i] > 0 ? a[i] : b[i];                                          \
    }                                                                                 \
  }
#endif

DECLARE_VML_WHERE_FUNCTION_IMPL(int32_t, float16)
DECLARE_VML_WHERE_FUNCTION_IMPL(int32_t, float)
DECLARE_VML_WHERE_FUNCTION_IMPL(int32_t, double)
DECLARE_VML_WHERE_FUNCTION_IMPL(int64_t, float16)
DECLARE_VML_WHERE_FUNCTION_IMPL(int64_t, float)
DECLARE_VML_WHERE_FUNCTION_IMPL(int64_t, double)

#undef DECLARE_VML_WHERE_FUNCTION_IMPL

}  // namespace blaze


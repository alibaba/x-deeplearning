/*
 * \file vml.h
 * \brief The VML routine on CPU Architecture
 */
#pragma once

#include "blaze/common/context.h"

#ifdef USE_MKL
#include "mkl.h"
#endif

namespace blaze {

#ifndef DECLARE_VML_FUNCTION
#define DECLARE_VML_FUNCTION(name)                                \
  template <typename T, class Context>                            \
  void name(const int N, const T* x, T* y, Context* context)
#endif

DECLARE_VML_FUNCTION(VML_Exp);
DECLARE_VML_FUNCTION(VML_Log);
DECLARE_VML_FUNCTION(VML_Cos);
DECLARE_VML_FUNCTION(VML_Acos);
DECLARE_VML_FUNCTION(VML_Sin);
DECLARE_VML_FUNCTION(VML_Asin);
DECLARE_VML_FUNCTION(VML_Tan);
DECLARE_VML_FUNCTION(VML_Atan);
DECLARE_VML_FUNCTION(VML_Tanh);
DECLARE_VML_FUNCTION(VML_Abs);
DECLARE_VML_FUNCTION(VML_Sqrt);

template <typename T, typename Context>
void VML_Powx(const int N, const T* a, T b, T* y, Context* context);

#ifndef DECLARE_VML_BINARY_OP_FUNCTION
#define DECLARE_VML_BINARY_OP_FUNCTION(name)                      \
  template <typename T, class Context>                            \
  void name(const int N, const T* a, const T* b, T* y, Context* context)
#endif

DECLARE_VML_BINARY_OP_FUNCTION(VML_Add);
DECLARE_VML_BINARY_OP_FUNCTION(VML_Sub);
DECLARE_VML_BINARY_OP_FUNCTION(VML_Mul);
DECLARE_VML_BINARY_OP_FUNCTION(VML_Div);

template <typename T, class Context>
void VML_Set(const int N, T* a, T v, Context* context);

template <typename DstT, class SrcT, class Context>
void VML_Set(const int N, DstT* dst, const SrcT* src, Context* context);

template <typename T1, typename T2, class Context>
void VML_Where(const int N, const T1* condition, const T2* a, const T2* b, T2* y, Context* context);

}  // namespace blaze

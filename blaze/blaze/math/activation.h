/*
 * \file activation.h
 * \brief The activation.
 */
#pragma once

#include "blaze/common/common_defines.h"

#include <math.h>

#include "blaze/common/types.h"

namespace blaze {

enum ActivationType {
  kRelu = 0,
  kPRelu,
  kDice,
  kTanh,
  kSigmoid,
  kBN,
};

template <ActivationType at>
struct Activation {
};

template <>
struct Activation<kRelu> {
  template <typename T>
  BLAZE_INLINE BLAZE_DEVICE void operator()(float alpha, const T* input, T* output) {
    *output = ((float)(*input) >= 0) ? *input : (T)(alpha * (float)(*input));
  }
};

template <>
struct Activation<kPRelu> {
  template <typename T, typename W>
  BLAZE_INLINE BLAZE_DEVICE void operator()(const T* input, const W* weight, T* output) {
    *output = (*input) * ((*input > 0) + ((*input < 0) * (*weight)));
  }
};

template <>
struct Activation<kDice> {
  template <typename T, typename W>
  BLAZE_INLINE BLAZE_DEVICE void operator()(const T* input, const W* gamma, const W* mean, const W* avr, T* output) {
    T x_normed = ((*input) - (*mean)) / sqrtf((*avr) + 1e-8);
    T x_p = 1.0 / (1.0 + expf(-x_normed));
    *output = (1 - x_p) * (*gamma) * (*input) + x_p * (*input);
  }
};

template <>
struct Activation<kTanh> {
  template <typename T>
  BLAZE_INLINE BLAZE_DEVICE void operator()(const T* input, T* output) {
    *output = tanh(*input);
  }
};

template <>
struct Activation<kSigmoid> {
  template <typename T>
  BLAZE_INLINE BLAZE_DEVICE void operator()(const T* input, T* output) {
    *output = 1. / (1. + expf(-(*input)));
  }
};

template <>
struct Activation<kBN> {
  template <typename T, typename W>
  BLAZE_INLINE BLAZE_DEVICE void operator()(const T* input, const W* gamma, const W* beta, const W* mean, const W* var, T* output) {
    T x_normed = ((*input) - (*mean)) / sqrtf((*var) + 1e-8);
    *output = (*gamma) * x_normed + (*beta);
  }
};

}  // namespace blaze


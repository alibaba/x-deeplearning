/*
 * \file binary_search.h
 * \desc Binary search algorithm
 */
#pragma once

#include "blaze/common/types.h"

#include <x86intrin.h>
#include <immintrin.h>
#include <f16cintrin.h>

#include "blaze/math/vml.h"

namespace blaze {

template <typename T>
void BinarySearch(const T *data,
                  int N,
                  const T *keys,
                  int M,
                  T *result);

}  // namespace blaze
/*
 * \file cross_common_defines.h 
 * \brief The cross common defines. 
 */
#pragma once

#include "blaze/common/murmurhash.h"

namespace blaze {

template <typename K_DType, typename V_DType>
void QuickSortByID(K_DType* id, V_DType* value, int l, int r) {
  if (l > r) return;
  int mid = (l + r) / 2;
  int i = l;
  int j = r;
  K_DType mid_value = id[mid];
  while (true) {
    while (id[i] < mid_value) i++;
    while (id[j] > mid_value) j--;
    if (i >= j)
      break;
    std::swap(id[i], id[j]);
    std::swap(value[i], value[j]);
  }
  id[j] = mid_value;
  QuickSortByID<K_DType>(id, value, l, j - 1);
  QuickSortByID<K_DType>(id, value, j + 1, r);
  return;
}

static const int kDotProductIdSizePerAd = 5;
static const char* kSumAa = "sum_aa";
static const char* kSumAb = "sum_ab";
static const char* kSumBb = "sum_bb";
static const char* kSumAbsAb = "sum_abs_ab";
static const char* kSumAbAb = "sum_abab";
static const uint64_t kSumAaHashID = blaze::MurmurHash64A(kSumAa, strlen(kSumAa));
static const uint64_t kSumAbHashID = blaze::MurmurHash64A(kSumAb, strlen(kSumAb));
static const uint64_t kSumBbHashID = blaze::MurmurHash64A(kSumBb, strlen(kSumBb));
static const uint64_t kSumAbsAbHashID = blaze::MurmurHash64A(kSumAbsAb, strlen(kSumAbsAb));
static const uint64_t kSumAbAbHashID = blaze::MurmurHash64A(kSumAbAb, strlen(kSumAbAb));

enum CombineProcessFuncType {
  kCombineProcessFuncIDFlag = 1,
  kCombineProcessFuncLogIDFlag,
  kCombineProcessFuncSumFlag,
  kCombineProcessFuncLogSumFlag,
  kCombineProcessFuncMaxFlag,
  kCombineProcessFuncLogMaxFlag,
  kCombineProcessFuncMinFlag,
  kCombineProcessFuncLogMinFlag,
  kCombineProcessFuncAvgFlag,
  kCombineProcessFuncLogAvgFlag,
  kCombineProcessFuncCosFlag,
  kCombineProcessFuncLogCosFlag,
  kCombineProcessFuncDotSumFlag,
  kCombineProcessFuncLogDotSumFlag,
  kCombineProcessFuncDotL1NormFlag,
  kCombineProcessFuncDotL2NormFlag
};

}  // namespace blaze

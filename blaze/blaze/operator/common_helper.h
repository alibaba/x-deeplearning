/*
 * \file common_helper.h 
 * \brief The fusion common helper utility.
 */
#pragma once

#include <unordered_map>

#include "blaze/common/proto_helper.h"

namespace blaze {

class CommonHelper {
 public:
  static size_t GetSliceAxis(const ArgumentHelper* arg);
  static size_t GetSliceStart(const ArgumentHelper* arg);
  static size_t GetSliceEnd(const ArgumentHelper* arg);
  static size_t GetReduceAxis(const ArgumentHelper* arg);
};

static const std::string kAttrIsElementWise = "is_element_wise";

//--- A usefull AttrMap used in OperatorSchema
#define INSTANTIATE_SET_ATTRIBUTE(T, field_name)               \
  void SetAttr(const std::string& key, T value) {              \
    const auto& iter = this->field_name.find(key);             \
    if (iter == this->field_name.end()) {                      \
      this->field_name[key] = value;                           \
    } else {                                                   \
      LOG_ERROR("duplicate key: %s", key.c_str());             \
    }                                                          \
  }

#define INSTANTIATE_GET_ATTRIBUTE(T, field_name)               \
  T GetAttr(const std::string& key, T default_value) {         \
    const auto& iter = this->field_name.find(key);             \
    if (iter == this->field_name.end()) {                      \
      return default_value;                                    \
    } else {                                                   \
      return iter->second;                                     \
    }                                                          \
  }

class AttrMap {
 public:
  INSTANTIATE_SET_ATTRIBUTE(float, f_kv_)
  INSTANTIATE_SET_ATTRIBUTE(double, f_kv_)
  INSTANTIATE_SET_ATTRIBUTE(bool, i_kv_)
  INSTANTIATE_SET_ATTRIBUTE(int8_t, i_kv_)
  INSTANTIATE_SET_ATTRIBUTE(int16_t, i_kv_)
  INSTANTIATE_SET_ATTRIBUTE(int, i_kv_)
  INSTANTIATE_SET_ATTRIBUTE(int64_t, i_kv_)
  INSTANTIATE_SET_ATTRIBUTE(uint8_t, i_kv_)
  INSTANTIATE_SET_ATTRIBUTE(uint16_t, i_kv_)
  INSTANTIATE_SET_ATTRIBUTE(size_t, i_kv_)
  INSTANTIATE_SET_ATTRIBUTE(std::string, s_kv_)

  INSTANTIATE_GET_ATTRIBUTE(float, f_kv_)
  INSTANTIATE_GET_ATTRIBUTE(double, f_kv_)
  INSTANTIATE_GET_ATTRIBUTE(bool, i_kv_)
  INSTANTIATE_GET_ATTRIBUTE(int8_t, i_kv_)
  INSTANTIATE_GET_ATTRIBUTE(int16_t, i_kv_)
  INSTANTIATE_GET_ATTRIBUTE(int, i_kv_)
  INSTANTIATE_GET_ATTRIBUTE(int64_t, i_kv_)
  INSTANTIATE_GET_ATTRIBUTE(uint8_t, i_kv_)
  INSTANTIATE_GET_ATTRIBUTE(uint16_t, i_kv_)
  INSTANTIATE_GET_ATTRIBUTE(size_t, i_kv_)
  INSTANTIATE_GET_ATTRIBUTE(std::string, s_kv_)

 protected:
  std::unordered_map<std::string, std::string> s_kv_;
  std::unordered_map<std::string, int64_t> i_kv_;
  std::unordered_map<std::string, float> f_kv_;
};

#undef INSTANTIATE_SET_ATTRIBUTE
#undef INSTANTIATE_GET_ATTRIBUTE

// Return number of elements from dims
size_t NElemFromDim(const TensorShape& shape);

static const std::string kOpArgNameParallelNum = "parallel_num";

// Tha max input size of Op
#ifndef kMaxInputSize
#define kMaxInputSize 128
#endif

static const float kBnEpsilon = 0.001;
static const float kDiceEpsilon = 1e-8;

static std::string kIndicatorPrefix = "indicator";
static std::string kMask = "mask";
static std::string kSparseFeatureSep = ".";

static const char* kIdSuffix = ".ids";
static const char* kValueSuffix = ".values";
static const char* kIdNumSuffix = ".segments";

// Get the indicator level
int GetIndicatorLevel(const std::string& indicator_name);
InputType GetSparseInputType(const std::string& input_name);
std::string GetSparseFeatureName(const std::string& input_name);

static int kMaxLevel = 99;

}  // namespace blaze

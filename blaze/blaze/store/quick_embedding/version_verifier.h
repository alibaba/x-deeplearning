/*!
 * \file version_verifier.h
 * \desc Verify embedding version
 */
#pragma once

#include "blaze/store/quick_embedding/serializable.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/types.h"

namespace blaze {
namespace store {

extern const uint32_t QuickEmbeddingImplVersion;

enum class DictValueType {
  unknown = -1,
  fp32 = 0,
  fp16 = 1,
  int8 = 2
};

#define SWITCHTYPE_DictValueType(vt, t, ...)  \
switch(vt) { \
case DictValueType::fp32: {typedef float t; __VA_ARGS__} break; \
case DictValueType::fp16: {typedef float16 t; __VA_ARGS__} break; \
case DictValueType::int8: {typedef int8_t t; __VA_ARGS__} break; \
default : LOG_ERROR("unsupported value: %d", static_cast<std::underlying_type<DictValueType>::type>(vt)); break;\
}

template<typename T>
struct DictValueTypeFromType {
  static DictValueType valueType() {
    return DictValueType::unknown;
  }
};

template<>
struct DictValueTypeFromType<float> {
  static DictValueType valueType() {
    return DictValueType::fp32;
  }
};

template<>
struct DictValueTypeFromType<float16> {
  static DictValueType valueType() {
    return DictValueType::fp16;
  }
};

template<>
struct DictValueTypeFromType<int8_t> {
  static DictValueType valueType() {
    return DictValueType::int8;
  }
};

class VersionVerifier : public Serializable {
 public:
  explicit VersionVerifier(DictValueType valueType = DictValueType::fp32);

  ~VersionVerifier() override;

  // get version
  uint32_t version() const {
    return version_;
  }

  // get implementation version of the QuickEmbeddingDict
  uint32_t impl_version() const;

  // valueType of this embedding dict
  DictValueType value_type() const;

  // return index header byte size
  uint64_t ByteArraySize() const override;

  // load from istream
  bool Load(std::istream *is) override;

  // dump to ostream
  bool Dump(std::ostream *os) const override;

 private:
  const uint32_t version_;
  DictValueType type_;
};

}  // namespace store
}  // namespace blaze


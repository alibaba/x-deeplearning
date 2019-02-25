/*!
 * \file version_verifier.cc
 * \desc Identify model version by header
 */

#include "blaze/store/quick_embedding/version_verifier.h"

namespace {
const char* kCheckCode = "ALIMAMA_MODEL_SERVING_EMBEDDING";
const uint32_t kCheckCodeLen = strlen(kCheckCode);
}  // namespace

namespace blaze {
namespace store {

const uint32_t QuickEmbeddingImplVersion = 2;

VersionVerifier::VersionVerifier(DictValueType valueType) :
    type_(valueType),
    version_(1000 * QuickEmbeddingImplVersion
                 + static_cast<std::underlying_type<DictValueType>::type>(valueType)) {

}

VersionVerifier::~VersionVerifier() {
}

uint32_t VersionVerifier::impl_version() const {
  return version_ / 1000;
}

DictValueType VersionVerifier::value_type() const {
  return type_;
}

uint64_t VersionVerifier::ByteArraySize() const {
  return strlen(kCheckCode) + sizeof(version_);
}

bool VersionVerifier::Load(std::istream* is) {
  // [STEP1]: load check code
  char check_code[kCheckCodeLen];
  is->read(check_code, kCheckCodeLen);
  if (!is->good()) return false;
  // [STEP2]: verify check code
  if (strncmp(check_code, kCheckCode, kCheckCodeLen) != 0) {
    return false;
  }
  // [STEP3]: load version
  is->read((char*)&version_, sizeof(version_));
  if (!is->good()) return false;
#define QED_SWITCHBW2INT(bw) case static_cast<std::underlying_type<DictValueType>::type>(DictValueType::bw):\
                               type_ = DictValueType::bw; break;
  switch (version_ % 1000) {
    QED_SWITCHBW2INT(fp32)
    QED_SWITCHBW2INT(fp16)
    QED_SWITCHBW2INT(int8)
    default:
      type_ = DictValueType::unknown;
  }
#undef QED_SWITCHBW2INT
  return true;
}

bool VersionVerifier::Dump(std::ostream* os) const {
  // [STEP1]: write check code
  os->write(kCheckCode, strlen(kCheckCode));
  if (!os->good()) return false;
  // [STEP2]: write version
  os->write((char*)&version_, sizeof(version_));
  return os->good();
}

}  // namespace store
}  // namespace blaze


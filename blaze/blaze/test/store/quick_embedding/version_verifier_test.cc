/*!
 * \file version_verifier_test.cc
 * \brief The version verifier test unit
 */
#include "blaze/store/quick_embedding/version_verifier.h"

#include <fstream>

#include "thirdparty/gtest/gtest.h"

namespace blaze {
namespace store {

TEST(TestVersionVerifier, impl_version) {
  VersionVerifier verifier;
  EXPECT_EQ(QuickEmbeddingImplVersion, verifier.impl_version());
}

TEST(TestVersionVerifier, version) {
  DictValueType vt = DictValueType::fp16;
  VersionVerifier verifier(vt);
  uint32_t expect_version = QuickEmbeddingImplVersion * 1000
                            + static_cast<std::underlying_type<DictValueType>::type>(vt);
  EXPECT_EQ(expect_version, verifier.version());
}

TEST(TestVersionVerifier, value_type) {
  DictValueType vt = DictValueType::fp16;
  VersionVerifier verifier(vt);
  EXPECT_EQ(vt, verifier.value_type());
}

TEST(TestVersionVerifier, Dump) {
  VersionVerifier verifier_fp32(DictValueType::fp32);
  std::ofstream out1("out.ut.verifier_fp32.bin", std::ios::binary);
  EXPECT_TRUE(out1.good());
  EXPECT_TRUE(verifier_fp32.Dump(&out1));
  out1.close();

  VersionVerifier verifier_fp16(DictValueType::fp16);
  std::ofstream out2("out.ut.verifier_fp16.bin", std::ios::binary);
  EXPECT_TRUE(out2.good());
  EXPECT_TRUE(verifier_fp16.Dump(&out2));
  out2.close();

  VersionVerifier verifier_int8(DictValueType::int8);
  std::ofstream out3("out.ut.verifier_int8.bin", std::ios::binary);
  EXPECT_TRUE(out3.good());
  EXPECT_TRUE(verifier_int8.Dump(&out3));
  out3.close();

  VersionVerifier verifier_unknown(DictValueType::unknown);
  std::ofstream out4("out.ut.verifier_unknown.bin", std::ios::binary);
  EXPECT_TRUE(out4.good());
  EXPECT_TRUE(verifier_unknown.Dump(&out4));
  out4.close();
}

TEST(TestVersionVerifier, Load) {
  VersionVerifier verifier;
  std::ifstream is1("out.ut.verifier_fp32.bin", std::ios::binary);
  EXPECT_TRUE(is1.good());
  EXPECT_TRUE(verifier.Load(&is1));
  EXPECT_EQ(verifier.value_type(), DictValueType::fp32);
  is1.close();

  std::ifstream is2("out.ut.verifier_fp16.bin", std::ios::binary);
  EXPECT_TRUE(is2.good());
  EXPECT_TRUE(verifier.Load(&is2));
  EXPECT_EQ(verifier.value_type(), DictValueType::fp16);
  is2.close();

  std::ifstream is3("out.ut.verifier_int8.bin", std::ios::binary);
  EXPECT_TRUE(is3.good());
  EXPECT_TRUE(verifier.Load(&is3));
  EXPECT_EQ(verifier.value_type(), DictValueType::int8);
  is3.close();

  std::ifstream is4("out.ut.verifier_unknown.bin", std::ios::binary);
  EXPECT_TRUE(is4.good());
  EXPECT_TRUE(verifier.Load(&is4));
  EXPECT_EQ(verifier.value_type(), DictValueType::unknown);
  is4.close();
}

TEST(TestVersionVerifier, ByteArraySize) {
  VersionVerifier verifier;
  uint64_t expect_size = 31 + sizeof(verifier.version());
  EXPECT_EQ(expect_size, verifier.ByteArraySize());
}

}  // namespace store
}  // namespace blaze

/*!
 * \file weight_blob_test.cc
 * \brief The weight blob test unit
 */
#include "blaze/store/quick_embedding/weight_blob.h"

#include <fstream>

#include "thirdparty/gtest/gtest.h"

namespace blaze {
namespace store {

static WeightBlob<float> fwb, load_fwb;

TEST(Test, AllocateMemory) {
  EXPECT_TRUE(fwb.AllocateMemory(30));
}

TEST(TestWeightBlob, InsertWeights) {
  float src[12];
  for (int i = 0; i < 12; ++i) {
    src[i] = (float) i;
  }
  float *dst = nullptr;
  uint64_t offset = fwb.InsertWeights(12, &dst);
  EXPECT_EQ(1, offset);
  EXPECT_TRUE(dst != nullptr);
  memcpy(dst, src, sizeof(float) * 12);

  float src2[18];
  for (int i = 0; i < 18; ++i) {
    src2[i] = (float) i;
  }

  offset = fwb.InsertWeights(18, &dst);
  EXPECT_EQ(1 + sizeof(float) * 12, offset);
  EXPECT_TRUE(dst != nullptr);
  memcpy(dst, src2, sizeof(float) * 18);
}

TEST(TestWeightBlob, GetWeights) {
  float *weights = fwb.GetWeights(1);
  for (int i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ((float) i, weights[i]);
  }
  weights = fwb.GetWeights(1 + sizeof(float) * 12);
  for (int i = 0; i < 18; ++i) {
    EXPECT_FLOAT_EQ((float) i, weights[i]);
  }
}

TEST(TestWeightBlob, Dump) {
  std::ofstream test_out("test.ut.weightblob.bin", std::ios::binary);
  WeightBlob<float> empty_blob;
  EXPECT_FALSE(empty_blob.Dump(&test_out));
  test_out.close();

  std::ofstream out("out.ut.weightblob.bin", std::ios::binary);
  EXPECT_TRUE(out.good());
  EXPECT_TRUE(fwb.Dump(&out));
  out.close();
}

TEST(TestWeightBlob, Load) {
  std::ifstream is("out.ut.weightblob.bin", std::ios::binary);
  EXPECT_TRUE(is.good());
  EXPECT_TRUE(load_fwb.Load(&is));
  is.close();
  // check data
  float *weights = load_fwb.GetWeights(1);
  for (int i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ((float) i, weights[i]);
  }
  weights = load_fwb.GetWeights(1 + sizeof(float) * 12);
  for (int i = 0; i < 18; ++i) {
    EXPECT_FLOAT_EQ((float) i, weights[i]);
  }
}

TEST(TestWeightBlob, ByteArraySize) {
  uint64_t expect_size = sizeof(uint64_t) + sizeof(float) * 30 + 1;
  EXPECT_EQ(expect_size, fwb.ByteArraySize());
  EXPECT_EQ(fwb.ByteArraySize(), load_fwb.ByteArraySize());
}

}  // namespace store
}  // namespace blaze


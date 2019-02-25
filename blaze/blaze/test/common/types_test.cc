/*
 * \file types_test.cc
 * \brief The types test module
 */
#include "gtest/gtest.h"

#include <math.h>

#include "blaze/common/types.h"

namespace blaze {

TEST(TestFloat16, Float16) {
  float a = 1.0;
  float16 a_16 = a;
  float a_b = a_16;
  LOG_INFO("a_b=%f", a_b);
  EXPECT_FLOAT_EQ(a_b, a);

  a = 2.0;
  a_16 = static_cast<float16>(a);
  a_b = a_16;
  LOG_INFO("a_b=%f", a_b);
  EXPECT_FLOAT_EQ(a_b, a);

  float z = a_16 + 1.0;
  float ez = expf(a_16);
}

TEST(TestDataTypeSize, All) {
  EXPECT_EQ(DataTypeSize(kFloat), sizeof(float));
  EXPECT_EQ(DataTypeSize(kFloat16), sizeof(float16));
  EXPECT_EQ(DataTypeSize(kInt32), sizeof(int32_t));
  EXPECT_EQ(DataTypeSize(kBool), sizeof(bool));
  EXPECT_EQ(DataTypeSize(kInt8), sizeof(int8_t));
  EXPECT_EQ(DataTypeSize(kUInt8), sizeof(uint8_t));
  EXPECT_EQ(DataTypeSize(kUInt16), sizeof(uint16_t));
  EXPECT_EQ(DataTypeSize(kInt16), sizeof(int16_t));
  EXPECT_EQ(DataTypeSize(kInt64), sizeof(int64_t));
}

}  // namespace blaze


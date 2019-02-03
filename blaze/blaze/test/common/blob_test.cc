/*
 * \file blob_test.cc
 * \brief The blob test module
 */
#include "gtest/gtest.h"

#include "blaze/common/blob.h"

namespace blaze {

TEST(TestBlob, Blob) {
  DeviceOption device_option;
  Blob blob1(device_option);
  EXPECT_EQ(blob1.as<char>(), nullptr);
  EXPECT_EQ(blob1.data(), nullptr);
  EXPECT_EQ(blob1.data_type(), DataType::kFloat);

  Blob blob2(device_option, { 2UL, 3UL }, kInt32);
  EXPECT_EQ(blob2.data_type(), DataType::kInt32);
  EXPECT_EQ(blob2.size(), 6);
  EXPECT_EQ(blob2.capacity(), 6);
}

TEST(TestBlob, Reshape) {
  DeviceOption device_option;
  Blob blob(device_option);
  blob.Reshape({ 2, 3 });
  EXPECT_EQ(blob.shape().size(), 2);
  EXPECT_EQ(blob.shape()[0], 2);
  EXPECT_EQ(blob.shape()[1], 3);

  blob.Reshape({ 3, 3 });
  EXPECT_EQ(blob.size(), 9);
  EXPECT_EQ(blob.capacity(), 9);

  blob.Reshape({ 2, 3 });
  EXPECT_EQ(blob.size(), 6);
  EXPECT_EQ(blob.capacity(), 9);
}

TEST(TestBlob, Copy) {
  DeviceOption device_option;
  Blob src_blob(device_option);
  Blob dst_blob(device_option);

  src_blob.Reshape({ 2, 3 });
  dst_blob.Reshape({ 2, 3 });
  for (TIndex k = 0; k < src_blob.size(); ++k) {
    src_blob.as<float>()[k] = 1.0;
  }

  Copy(&dst_blob, &src_blob, 0);
  for (TIndex k = 0; k < dst_blob.size(); ++k) {
    EXPECT_FLOAT_EQ(dst_blob.as<float>()[k], 1.0);
  }
}

}  // namespace blaze



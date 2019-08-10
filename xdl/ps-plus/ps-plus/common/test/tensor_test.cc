#include "gtest/gtest.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/initializer/constant_initializer.h"

using ps::TensorShape;
using ps::DataType;
using ps::Tensor;
using ps::DataType;
using ps::Initializer;
using ps::initializer::ConstantInitializer;

TEST(TensorTest, Contructor) {
  TensorShape shape({10, 5});
  Tensor x(DataType::kInt8, shape, new ConstantInitializer(1));
  Tensor y(DataType::kInt8, shape, new ConstantInitializer(1));
  for (int i = 0; i < 50; i++) {
    x.Raw<int8_t>()[i] = i;
    y.Raw<int8_t>()[i] = i + 10;
  }
  Tensor z(std::move(y));
  EXPECT_EQ(TensorShape({10, 5}), z.Shape());
  EXPECT_EQ(DataType::kInt8, z.Type());
  for (int i = 0; i < 50; i++) {
    EXPECT_EQ(i + 10, z.Raw<int8_t>()[i]);
  }

  char buf[8] = {0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78};
  Tensor k(DataType::kInt8, TensorShape({4, 8}), buf, new ConstantInitializer(1));
  EXPECT_EQ(0x7867564534231201, k.Raw<int64_t>()[0]);
}

TEST(TensorTest, Initializer) {
  TensorShape shape({4, 8});
  Tensor x(DataType::kInt8, shape, new ConstantInitializer(1), Tensor::TType::kSegment, true);
  EXPECT_EQ(0x0101010101010101, x.Raw<int64_t>()[0]);
  EXPECT_EQ(0x0101010101010101, x.Raw<int64_t>()[1]);
  EXPECT_EQ(0x0101010101010101, x.Raw<int64_t>()[2]);
  EXPECT_EQ(0x0101010101010101, x.Raw<int64_t>()[3]);

  for (size_t i = 0; i < 32; i++) {
    x.Raw<int8_t>()[i] = i;
  }
  EXPECT_EQ(0x0706050403020100, x.Raw<int64_t>()[0]);
  EXPECT_EQ(0x0F0E0D0C0B0A0908, x.Raw<int64_t>()[1]);
  EXPECT_EQ(0x1716151413121110, x.Raw<int64_t>()[2]);
  EXPECT_EQ(0x1F1E1D1C1B1A1918, x.Raw<int64_t>()[3]);

  x.ReShape(TensorShape({2, 8}));
  EXPECT_EQ(0x0706050403020100, x.Raw<int64_t>()[0]);
  EXPECT_EQ(0x0F0E0D0C0B0A0908, x.Raw<int64_t>()[1]);

  x.ReShape(TensorShape({4, 8}));
  EXPECT_EQ(0x0706050403020100, x.Raw<int64_t>()[0]);
  EXPECT_EQ(0x0F0E0D0C0B0A0908, x.Raw<int64_t>()[1]);
  EXPECT_EQ(0x1716151413121110, x.Raw<int64_t>()[2]);
  EXPECT_EQ(0x1F1E1D1C1B1A1918, x.Raw<int64_t>()[3]);

  for (size_t i = 0; i < 32; i++) {
    x.Raw<int8_t>()[i] = i;
  }
  EXPECT_EQ(0x0706050403020100, x.Raw<int64_t>()[0]);
  EXPECT_EQ(0x0F0E0D0C0B0A0908, x.Raw<int64_t>()[1]);
  EXPECT_EQ(0x1716151413121110, x.Raw<int64_t>()[2]);
  EXPECT_EQ(0x1F1E1D1C1B1A1918, x.Raw<int64_t>()[3]);

  x.ClearId(2);
  EXPECT_EQ(0x0706050403020100, x.Raw<int64_t>()[0]);  
  EXPECT_EQ(0x0F0E0D0C0B0A0908, x.Raw<int64_t>()[1]);
  EXPECT_EQ(0x0101010101010101, x.Raw<int64_t>()[2]);
  EXPECT_EQ(0x1F1E1D1C1B1A1918, x.Raw<int64_t>()[3]);
}

TEST(TensorTest, CopyAndMoveForOwnBuffer) {
  TensorShape shape({1, 8});
  Tensor x(DataType::kInt8, shape, new ConstantInitializer(1));
  EXPECT_EQ(0x0101010101010101, x.Raw<int64_t>()[0]);

  Tensor y(x);
  EXPECT_EQ(0x0101010101010101, y.Raw<int64_t>()[0]);

  Tensor z(std::move(x));
  EXPECT_EQ(0x0101010101010101, z.Raw<int64_t>()[0]);

  Tensor k;
  k = y;
  EXPECT_EQ(0x0101010101010101, k.Raw<int64_t>()[0]);

  k = std::move(y);
  EXPECT_EQ(0x0101010101010101, k.Raw<int64_t>()[0]);
}

TEST(TensorTest, CopyAndMoveForNotOwnBuffer) {
  char buf[8] = {0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78};
  Tensor x(DataType::kInt8, TensorShape({4, 8}), buf, new ConstantInitializer(1));
  EXPECT_EQ(0x7867564534231201, x.Raw<int64_t>()[0]);

  Tensor y(x);
  EXPECT_EQ(0x7867564534231201, y.Raw<int64_t>()[0]);

  Tensor z(std::move(x));
  EXPECT_EQ(0x7867564534231201, z.Raw<int64_t>()[0]);

  Tensor k;
  k = y;
  EXPECT_EQ(0x7867564534231201, k.Raw<int64_t>()[0]);

  k = std::move(y);
  EXPECT_EQ(0x7867564534231201, k.Raw<int64_t>()[0]);
}


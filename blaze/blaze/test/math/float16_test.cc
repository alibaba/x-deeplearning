/*
 * \file float16_test.cc
 * \brief The float16 test unit
 */
#include "gtest/gtest.h"

#include <math.h>

#include "blaze/math/float16.h"

namespace blaze {

TEST(TestFloat2Half, Float2Half) {
  float s = 0.0012521;
  float16 h;
  float2half(&s, 1, &h);
  float s_back;
  half2float(&h, 1, &s_back);
  EXPECT_TRUE(fabs(s - s_back) <= 1e-6);
  
  float array[12];
  for (int i = 0; i < 12; i++) {
    array[i] = 0.001 + i * 0.001;
  }
  float16 array_h[12];
  float2half(array, 12, array_h);
  float array_back[12];
  half2float(array_h, 12, array_back);

  for (int i = 0; i < 12; ++i) {
    EXPECT_TRUE(fabs(array[i] - array_back[i]) <= 1e-5);
    LOG_INFO("%f -> %f", array[i], array_back[i]);
  }
}

void InitInput(float16* a, float16* b, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = float16(0.001 * i);
    b[i] = float16(1.01 * i);
  }
}

TEST(TestVML, Add) {
  float16 a[12], b[12], c[12];
  InitInput(a, b, 12);
  VML_Add<float16, CPUContext>(12, a, b, c, nullptr);
  for (int i = 0; i < 12; ++i) {
    EXPECT_TRUE(fabs((float)c[i] - (0.001 * i + 1.01 * i)) <= 1e-2);
    LOG_INFO("Range: %f %f", (float)c[i], (0.001 * i + 1.01 * i));
  }
}

TEST(TestVML, Sub) {
  float16 a[12], b[12], c[12];
  InitInput(a, b, 12);
  VML_Sub<float16, CPUContext>(12, a, b, c, nullptr);
  for (int i = 0; i < 12; ++i) {
    EXPECT_TRUE(fabs((float)c[i] - (0.001 * i - 1.01 * i)) <= 1e-2);
    LOG_INFO("Range: %f %f", (float)c[i], (0.001 * i - 1.01 * i));
  }
}

TEST(TestVML, Mul) {
  float16 a[12], b[12], c[12];
  InitInput(a, b, 12);
  VML_Mul<float16, CPUContext>(12, a, b, c, nullptr);
  for (int i = 0; i < 12; ++i) {
    EXPECT_TRUE(fabs((float)c[i] - (0.001 * i * 1.01 * i)) <= 1e-4);
    LOG_INFO("Range: %f %f", (float)c[i], (0.001 * i * 1.01 * i));
  }
}

TEST(TestVML, Div) {
  float16 a[12], b[12], c[12];
  InitInput(a, b, 12);
  VML_Div<float16, CPUContext>(12, a, b, c, nullptr);
  for (int i = 1; i < 12; ++i) {
    EXPECT_TRUE(fabs((float)c[i] - ((0.001 * i) / (1.01 * i))) <= 1e-4);
    LOG_INFO("Range: %f %f", (float)c[i], ((0.001 * i) / (1.01 * i)));
  }
}

TEST(TestAdd, Add) {
  float16 a = 0.001;
  float16 b = 0.002;
  float16 c = a + b;
  EXPECT_TRUE(fabs((float)c - 0.003) <= 1e-3);
}

TEST(TestSub, Sub) {
  float16 a = 0.001;
  float16 b = 0.002;
  float16 c = a - b;
  EXPECT_TRUE(fabs((float)c - (-0.001)) <= 1e-3);
}

TEST(TestMul, Mul) {
  float16 a = 0.001;
  float16 b = 0.002;
  float16 c = a * b;
  EXPECT_TRUE(fabs((float)c - 0.00002) <= 1e-3);
}

TEST(TestDiv, Div) {
  float16 a = 0.001;
  float16 b = 0.002;
  float16 c = a / b;
  EXPECT_TRUE(fabs((float)c - 0.5) <= 1e-4);
}

}  // namespace blaze


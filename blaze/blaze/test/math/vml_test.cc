/*
 * \file vml_test.cc
 * \brief The vml test unit
 */
#include "gtest/gtest.h"

#include <math.h>

#include "blaze/math/vml.h"
#include "blaze/math/float16.h"

namespace blaze {

#define N 100

template <typename T>
void ExpTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Exp<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], expf(0.01 * i));
  }
}

TEST(TestExp, Exp) {
  ExpTest<float>();
  ExpTest<double>();
}

template <typename T>
void LogTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i + 1;
  }
  T y[N];
  VML_Log<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(fabs(y[i] - logf(0.01 * i + 1)) <= 1e-6);
  }
}

TEST(TestLog, Log) {
  LogTest<float>();
  LogTest<double>();
}

template <typename T>
void CosTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Cos<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], cosf(0.01 * i));
  }
}

TEST(TestCos, Cos) {
  CosTest<float>();
  CosTest<double>();
}

template <typename T>
void AcosTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Acos<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(fabs(y[i] - acosf(0.01 * i)) <= 1e-6);
  }
}

TEST(TestAcos, Acos) {
  AcosTest<float>();
  AcosTest<double>();
}

template <typename T>
void SinTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Sin<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], sinf(0.01 * i));
  }
}

TEST(TestSin, Sin) {
  SinTest<float>();
  SinTest<double>();
}

template <typename T>
void AsinTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Asin<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], asinf(0.01 * i));
  }
}

TEST(TestAsin, Asin) {
  AsinTest<float>();
  AsinTest<double>();
}

template <typename T>
void TanTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Tan<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], tanf(0.01 * i));
  }
}

TEST(TestTan, Tan) {
  TanTest<float>();
  TanTest<double>();
}

template <typename T>
void TanhTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Tanh<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], tanhf(0.01 * i));
  }
}

TEST(TestTanh, Tanh) {
  TanhTest<float>();
  TanhTest<double>();
}

template <typename T>
void AtanTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Atan<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], atanf(0.01 * i));
  }
}

TEST(TestAtan, Atan) {
  AtanTest<float>();
  AtanTest<double>();
}

template <typename T>
void AbsTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = -0.01 * i;
  }
  T y[N];
  VML_Abs<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], fabs(-0.01 * i));
  }
}

TEST(TestAbs, Abs) {
  AbsTest<float>();
  AbsTest<double>();
}

template <typename T>
void SqrtTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Sqrt<T, CPUContext>(N, x, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], sqrtf(0.01 * i));
  }
}

TEST(TestSqrt, Sqrt) {
  SqrtTest<float>();
  SqrtTest<double>();
}

template <typename T>
void PowxTest() {
  T x[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0.01 * i;
  }
  T y[N];
  VML_Powx<T, CPUContext>(N, x, 2.5, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], powf(0.01 * i, 2.5));
  }
}

TEST(TestPowx, Powx) {
  PowxTest<float>();
  PowxTest<double>();
}

template <typename T>
void AddTest() {
  T a[N];
  T b[N];
  for (int i = 0; i < N; ++i) {
    a[i] = 0.01 * i;
    b[i] = 0.02 * i;
  }
  T y[N];
  VML_Add<T, CPUContext>(N, a, b, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], 0.03 * i);
  }
}

TEST(TestAdd, Add) {
  AddTest<float>();
  AddTest<double>();
}

template <typename T>
void SubTest() {
  T a[N];
  T b[N];
  for (int i = 0; i < N; ++i) {
    a[i] = 0.01 * i;
    b[i] = 0.02 * i;
  }
  T y[N];
  VML_Sub<T, CPUContext>(N, a, b, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], -0.01 * i);
  }
}

TEST(TestSub, Sub) {
  SubTest<float>();
  SubTest<double>();
}

template <typename T>
void MulTest() {
  T a[N];
  T b[N];
  for (int i = 0; i < N; ++i) {
    a[i] = 0.01 * i;
    b[i] = 0.02 * i;
  }
  T y[N];
  VML_Mul<T, CPUContext>(N, a, b, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], 0.01 * i * 0.02 * i);
  }
}

TEST(TestMul, Mul) {
  MulTest<float>();
  MulTest<double>();
}

template <typename T>
void DivTest() {
  T a[N];
  T b[N];
  for (int i = 0; i < N; ++i) {
    a[i] = 0.01 * i + 1;
    b[i] = 0.02 * i + 1;
  }
  T y[N];
  VML_Div<T, CPUContext>(N, a, b, y, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(y[i], (0.01 * i + 1) / (0.02 * i + 1));
  }
}

TEST(TestDiv, Div) {
  DivTest<float>();
  DivTest<double>();
}

template <typename T>
void SetTest() {
  T a[N];
  VML_Set<T, CPUContext>(N, a, 1.0, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(a[i], 1.0);
  }
}

TEST(TestSet, Set) {
  SetTest<float16>();
  SetTest<float>();
  SetTest<double>();
}

template <typename DstT, typename SrcT>
void Set2Test() {
  DstT b[N];
  SrcT a[N];
  for (int i = 0; i < N; ++i) {
    a[i] = 0.01 * i;
  }
  VML_Set<DstT, SrcT, CPUContext>(N, b, a, nullptr);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(static_cast<DstT>(a[i]), b[i]);
  }
};

TEST(TestSet2, Set) {
  Set2Test<float, float16>();
  Set2Test<float16, float>();
  Set2Test<float16, float16>();
  Set2Test<float, float>();
}

template <typename IType, typename T>
void WhereTest() {
  IType condition[N];
  T x[N];
  T y[N];
  T z[N];
  for (int i = 0; i < N; ++i) {
    condition[i] = i % 3;
    x[i] = 0.01 * i + 1;
    y[i] = 0.02 * i + 1;
  }
  VML_Where<IType, T, CPUContext>(N, condition, x, y, z, nullptr);
  for (int i = 0; i < N; ++i) {
    if (i % 3 == 0) {
      EXPECT_FLOAT_EQ(y[i], z[i]);
    } else {
      EXPECT_FLOAT_EQ(x[i], z[i]);
    }
  }
};

TEST(TestWhere, Where) {
  WhereTest<int32_t, float16>();
  WhereTest<int32_t, float>();
  WhereTest<int32_t, double>();
  WhereTest<int64_t, float16>();
  WhereTest<int64_t, float>();
  WhereTest<int64_t, double>();
}

}  // namespace blaze


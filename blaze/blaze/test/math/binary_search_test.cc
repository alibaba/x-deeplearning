/*
 * \file binary_search_test.cc
 * \brief The binary search test unit
 */
#include "gtest/gtest.h"

#include "blaze/math/binary_search.h"
#include "blaze/common/timer.h"

#define N_SIZE 1000
#define M_SIZE 400
#define LOOP_TIME 10000

namespace blaze {

template<typename T>
void SimpleBinarySearch(const T *data,
                        int N,
                        const T *keys,
                        int M,
                        T *result) {
  for (int i = 0; i < M; i++) {
    int left = 0;
    int right = N - 1;
    result[i] = -1;
    while (left <= right) {
      int middle = (left + right) / 2;
      if (keys[i] == data[middle]) {
        result[i] = middle;
        break;
      } else if (keys[i] > data[middle]) {
        left = middle + 1;
      } else {
        right = middle - 1;
      }
    }
  }
}

TEST(TestBinarySearch, BinarySearch) {
  int data[N_SIZE];
  for (int i = 0; i < N_SIZE; i++) {
    data[i] = i * 10;
    // std::cout << "data = " << data[i] << std::endl;
  }

  int key[M_SIZE];
  for (int i = 0; i < M_SIZE; i++) {
    key[i] = rand() % (N_SIZE * 10);
    // std::cout << "key[" << i << "] = " << key[i] << std::endl;
  }

  Timer timer;
  // 32bit integer simple test
  int simple_result[M_SIZE];
  timer.Start();
  for (int loop = 0; loop < LOOP_TIME; loop++) {
    SimpleBinarySearch<int>(data, N_SIZE, key, M_SIZE, simple_result);
  }
  timer.Stop();
  LOG_INFO("32bit Simple Time = %f", timer.GetElapsedTime());

  // 32bit integer AVX128 SIMD test
  int simd_result[M_SIZE];
  timer.Start();
  for (int loop = 0; loop < LOOP_TIME; loop++) {
    BinarySearch<int32_t>(data, N_SIZE, key, M_SIZE, simd_result);
  }
  timer.Stop();
  LOG_INFO("32bit AVX128 SIMD Time = %f", timer.GetElapsedTime());

  // check 32bit result
  for (int i = 0; i < M_SIZE; i++) {
    EXPECT_EQ(simple_result[i], simd_result[i]);
  }

  int64_t data_64bit[N_SIZE];
  for (int i = 0; i < N_SIZE; i++) {
    data_64bit[i] = i * 10;
    // std::cout << "data = " << data[i] << std::endl;
  }

  int64_t key_64bit[M_SIZE];
  for (int i = 0; i < M_SIZE; i++) {
    key_64bit[i] = rand() % (N_SIZE * 10);
    // std::cout << "key[" << i << "] = " << key[i] << std::endl;
  }

  // 64bit integer simple test
  int64_t simple_result_64bit[M_SIZE];
  timer.Start();
  for (int loop = 0; loop < LOOP_TIME; loop++) {
    SimpleBinarySearch<int64_t>(data_64bit, N_SIZE, key_64bit, M_SIZE, simple_result_64bit);
  }
  timer.Stop();
  LOG_INFO("64bit Simple Time = %f", timer.GetElapsedTime());

  // 64bit integer AVX256 SIMD test
  int64_t simd_result_64bit[M_SIZE];
  timer.Start();
  for (int loop = 0; loop < LOOP_TIME; loop++) {
    BinarySearch<int64_t>(data_64bit, N_SIZE, key_64bit, M_SIZE, simd_result_64bit);
  }
  timer.Stop();
  LOG_INFO("64bit AVX256 SIMD Time = %f", timer.GetElapsedTime());

  // check 64bit result
  for (int i = 0; i < M_SIZE; i++) {
    EXPECT_EQ(simple_result_64bit[i], simd_result_64bit[i]);
  }
}

}  // namespace blaze
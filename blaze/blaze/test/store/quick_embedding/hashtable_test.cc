/*!
 * \file hashtable_test.cc
 * \brief The hash table test unit
 */
#include "blaze/store/quick_embedding/hashtable.h"

#include <fstream>
#include <string>

#include "thirdparty/gtest/gtest.h"

namespace blaze {
namespace store {

const int kTestCaseNum = 40;
const int kOffset = 97;
static BulkLoadHashTable mht;
static HashTable load_hashtable;

TEST(TestHashTable, PreInsert) {
  for (int key = 0; key < kTestCaseNum; key++) {
    EXPECT_TRUE(mht.PreInsert(key, key));
  }
  for (int key = kOffset; key < kOffset + kTestCaseNum; key++) {
    EXPECT_TRUE(mht.PreInsert(key, key));
  }
}

TEST(TestHashTable, BulkLoad) {
  EXPECT_TRUE(mht.BulkLoad());
}

TEST(TestHashTable, Lookup) {
  for (int key = 0; key < kTestCaseNum; key++) {
    HashTable::Value value;
    mht.Lookup(key, &value);
    EXPECT_EQ(key, value);
  }
  for (int key = kOffset; key < kOffset + kTestCaseNum; key++) {
    HashTable::Value value;
    mht.Lookup(key, &value);
    EXPECT_EQ(key, value);
  }
}

TEST(TestHashTable, AvgLength) {
  EXPECT_FLOAT_EQ(2.0f, mht.AvgLength());
}

TEST(TestHashTable, Dump) {
  std::ofstream out("out.ut.hashtable.bin", std::ios::binary);
  EXPECT_TRUE(out.good());
  EXPECT_TRUE(mht.Dump(&out));
  out.close();
}

TEST(TestHashTable, Load) {
  std::ifstream is("out.ut.hashtable.bin", std::ios::binary);
  EXPECT_TRUE(is.good());
  EXPECT_TRUE(load_hashtable.Load(&is));
  is.close();
  // check data
  for (int key = 0; key < kTestCaseNum; key++) {
    HashTable::Value value;
    load_hashtable.Lookup(key, &value);
    EXPECT_EQ(key, value);
  }

  for (int i = kOffset; i < kOffset + kTestCaseNum; i++) {
    HashTable::Value value;
    load_hashtable.Lookup(i, &value);
    EXPECT_EQ(i, value);
  }
}

TEST(TestHashTable, ByteArraySize) {
  uint64_t expect_byte_size = sizeof(uint32_t) + sizeof(uint64_t) * 97
                              + sizeof(uint64_t) + (sizeof(uint64_t) + sizeof(uint64_t)) * kTestCaseNum * 2;
  EXPECT_EQ(expect_byte_size, mht.ByteArraySize());
  EXPECT_EQ(mht.ByteArraySize(), load_hashtable.ByteArraySize());
}

}  // namespace store
}  // namespace blaze

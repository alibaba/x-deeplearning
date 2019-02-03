/*!
 * \file trie_test.cc
 * \brief The trie test unit
 */
#include "blaze/store/quick_embedding/trie.h"

#include <fstream>

#include "thirdparty/gtest/gtest.h"

#define protected public

namespace blaze {
namespace store {

static BulkLoadTrie blt;
static Trie trie;

TEST(TestBulkLoadTrie, PreInsert) {
  // Non-ascii code character, return false
  EXPECT_FALSE(blt.PreInsert("中国", 1));
  // common pre insert
  EXPECT_TRUE(blt.PreInsert("101", 1));
  EXPECT_TRUE(blt.PreInsert("102", 2));
  EXPECT_TRUE(blt.PreInsert("801_1", 3));
  EXPECT_TRUE(blt.PreInsert("803", 4));
  // duplicate key & value, warn
  EXPECT_TRUE(blt.PreInsert("803", 4));
  // conflict key & value error
  EXPECT_FALSE(blt.PreInsert("803", 5));
}

TEST(TestBulkLoadTrie, BulkLoad) {
  EXPECT_TRUE(blt.BulkLoad());
}

TEST(TestTrie, Lookup) {
  Trie::Value value;
  // Non-ascii code character, return false
  EXPECT_FALSE(blt.Lookup("中国", &value));
  // common lookup
  EXPECT_TRUE(blt.Lookup("101", &value));
  EXPECT_EQ(1, value);
  EXPECT_TRUE(blt.Lookup("102", &value));
  EXPECT_EQ(2, value);
  EXPECT_TRUE(blt.Lookup("801_1", &value));
  EXPECT_EQ(3, value);
  EXPECT_TRUE(blt.Lookup("803", &value));
  EXPECT_EQ(4, value);
  EXPECT_FALSE(blt.Lookup("201", &value));

}

TEST(TestBulkLoadTrie, Dump) {
  std::ofstream test_out("test.ut.trie.bin", std::ios::binary);
  BulkLoadTrie empty_trie;
  EXPECT_FALSE(empty_trie.Dump(&test_out));
  test_out.close();

  std::ofstream out("out.ut.trie.bin", std::ios::binary);
  EXPECT_TRUE(out.good());
  EXPECT_TRUE(blt.Dump(&out));
  out.close();
}

TEST(TestTrie, Load) {
  Trie::Value value;
  // uninitialized, return false
  EXPECT_FALSE(trie.Lookup("101", &value));

  std::ifstream is("out.ut.trie.bin", std::ios::binary);
  EXPECT_TRUE(is.good());
  EXPECT_TRUE(trie.Load(&is));
  is.close();
  // check data

  EXPECT_TRUE(trie.Lookup("101", &value));
  EXPECT_EQ(1, value);
  EXPECT_TRUE(trie.Lookup("102", &value));
  EXPECT_EQ(2, value);
  EXPECT_TRUE(trie.Lookup("801_1", &value));
  EXPECT_EQ(3, value);
  EXPECT_TRUE(trie.Lookup("803", &value));
  EXPECT_EQ(4, value);
  EXPECT_FALSE(trie.Lookup("201", &value));
}

TEST(TestTrie, ByteArraySize) {
  EXPECT_EQ(blt.ByteArraySize(), trie.ByteArraySize());
}

}  // namespace store
}  // namespace blaze


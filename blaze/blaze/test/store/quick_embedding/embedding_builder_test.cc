/*!
 * \file embedding_builder_test.cc
 * \brief The embedding builder test unit
 */
#include "blaze/store/quick_embedding/embedding_builder.h"
#include "blaze/store/quick_embedding/quick_embedding_dict.h"

#include "thirdparty/gtest/gtest.h"

namespace blaze {
namespace store {

TEST(TestEmbeddingBuilder, Build) {
  EmbeddingBuilder<float> builder;
  const std::string path = "./utest_data/store/quick_embedding/";
  const std::string meta = "meta";
  builder.Build(path, meta, "output", 2);

  QuickEmbeddingDict dict;
  Status status = dict.Load("output");
  EXPECT_EQ(kOK, status);
}

}  // namespace store
}  // namespace blaze
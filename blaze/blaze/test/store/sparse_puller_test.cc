/*
 * \file sparse_puller_test.cc
 * \brief The sparse puller test unit
 */
#include "gtest/gtest.h"

#include <math.h>

#include "blaze/store/sparse_puller.h"

namespace blaze {
namespace store {

class TestSparsePuller : public SparsePuller {
 public:
  Status Load(const std::string& url) { return kOK; }
  Status Get(const std::vector<SparsePullerInput>& in,
             std::vector<SparsePullerOutput>& out) {
    return kOK;
  }
};

SparsePuller* CreateTestSparsePuller() {
  return new TestSparsePuller();
}

REGISTER_SPARSE_PULLER_CREATION("test", CreateTestSparsePuller);

TEST(TestSparsePuller, CreateSparsePuller) {
  SparsePuller* store = SparsePullerCreationRegisterer::Get()->CreateSparsePuller("local_store");
  EXPECT_TRUE(store == nullptr);

  SparsePuller* test_store = SparsePullerCreationRegisterer::Get()->CreateSparsePuller("test");
  EXPECT_TRUE(test_store != nullptr);
  EXPECT_TRUE(test_store->Load("") == kOK);
  delete test_store;
}

}  // namespace store
}  // namespace blaze


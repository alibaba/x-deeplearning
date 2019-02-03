/*
 * \file typeid_test.cc
 * \brief The typeid test
 */
#include "gtest/gtest.h"

#include "blaze/common/typeid.h"
#include "blaze/common/log.h"

namespace blaze {

class Foo {
 public:
  Foo(int x) : x_(x) { }

 protected:
  int x_;
};

TEST(TestTypeId, Foo) {
  const char* foo_name = DemangleType<Foo>();
  EXPECT_STREQ(foo_name, "blaze::Foo");
  LOG_INFO("foo_name=%s", foo_name);
}

}  // namespace blaze


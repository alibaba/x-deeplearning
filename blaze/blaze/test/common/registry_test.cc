/*
 * \file registry_test.cc
 * \brief The registry test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/common/registry.h"
#include "blaze/common/log.h"

namespace blaze {

class Foo {
 public:
  explicit Foo(int x) { LOG_INFO("Foo"); }
};

DECLARE_REGISTRY(FooRegistry, Foo, int);
DEFINE_REGISTRY(FooRegistry, Foo, int);

class Bar : public Foo {
 public:
  explicit Bar(int x) : Foo(x) { LOG_INFO("Bar"); }
};
REGISTER_CLASS(FooRegistry, Bar, Bar);

class AnotherBar : public Foo {
 public:
  explicit AnotherBar(int x) : Foo(x) { LOG_INFO("Another Bar"); }
};
REGISTER_CLASS(FooRegistry, AnotherBar, AnotherBar);

TEST(TestRegistry, CanRunCreator) {
  std::unique_ptr<Foo> bar(FooRegistry()->Create("Bar", 1));
  EXPECT_TRUE(bar != nullptr) << "Cannot create bar";
  std::unique_ptr<Foo> another_bar(FooRegistry()->Create("AnotherBar", 1));
  EXPECT_TRUE(another_bar != nullptr);
}

TEST(TestRegistry, ReturnNullOnNonExistingCreator) {
  EXPECT_EQ(FooRegistry()->Create("Non-Existing Bar", 1), nullptr);
}

}  // namespace blaze

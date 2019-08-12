#include "gtest/gtest.h"
#include "ps-plus/common/data.h"

using ps::Data;
using ps::WrapperData;

namespace {

class MockData {
 public:
  MockData() {
  }
  MockData(int x) {
    init_type = 0;
  }
  MockData(const MockData&) {
    init_type = 1;
  }
  MockData(MockData&&) {
    init_type = 2;
  }
  int init_type;
};

}

TEST(DataTest, WrapperData) {
  MockData d;
  WrapperData<MockData>* d1 = new WrapperData<MockData>(1);
  WrapperData<MockData>* d2 = new WrapperData<MockData>(d);
  WrapperData<MockData>* d3 = new WrapperData<MockData>(std::move(d1->Internal()));
  EXPECT_EQ(0, d1->Internal().init_type);
  EXPECT_EQ(1, d2->Internal().init_type);
  EXPECT_EQ(2, d3->Internal().init_type);
}


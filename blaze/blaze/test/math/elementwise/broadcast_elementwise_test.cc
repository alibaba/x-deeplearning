/*
 * \file broadcast_elementwise_test.cc
 * \brief The broadcast elementwise test unit
 */

#include <vector>
#include "gtest/gtest.h"

#include "blaze/math/elementwise/broadcast_elementwise.h"

using std::vector;

namespace blaze {
namespace broadcast {

TEST(TestBroadcastElementwise, BroadcastShapeCompact) {
  vector<TIndex> lshape = {3, 1, 2};
  vector<TIndex> rshape = {2, 1};
  vector<TIndex> oshape = {3, 2, 2};
  vector<TIndex> new_lshape;
  vector<TIndex> new_rshape;
  vector<TIndex> new_oshape;
  int nDim = BroadcastShapeCompact(lshape, rshape, oshape,
      &new_lshape, &new_rshape, &new_oshape);
  EXPECT_EQ(3u, nDim);
  vector<TIndex> expected_lshape = {3, 1, 2};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_lshape[i], new_lshape[i]); 
  }
  vector<TIndex> expected_rshape = {1, 2, 1};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_rshape[i], new_rshape[i]); 
  }
  vector<TIndex> expected_oshape = {3, 2, 2};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_oshape[i], new_oshape[i]); 
  }
}

TEST(TestBroadcastElementwise, BroadcastShapeCompact_1) {
  vector<TIndex> lshape = {3, 2, 2};
  vector<TIndex> rshape = {2, 1};
  vector<TIndex> oshape = {3, 2, 2};
  vector<TIndex> new_lshape;
  vector<TIndex> new_rshape;
  vector<TIndex> new_oshape;
  int nDim = broadcast::BroadcastShapeCompact(lshape, rshape, oshape,
      &new_lshape, &new_rshape, &new_oshape);
  EXPECT_EQ(3u, nDim);
  vector<TIndex> expected_lshape = {3, 2, 2};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_lshape[i], new_lshape[i]); 
  }
  vector<TIndex> expected_rshape = {1, 2, 1};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_rshape[i], new_rshape[i]); 
  }
  vector<TIndex> expected_oshape = {3, 2, 2};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_oshape[i], new_oshape[i]); 
  }
}

TEST(TestBroadcastElementwise, BroadcastShapeCompact_2) {
  vector<TIndex> lshape = {3, 2, 1};
  vector<TIndex> rshape = {3, 2, 2};
  vector<TIndex> oshape = {3, 2, 2};
  vector<TIndex> new_lshape;
  vector<TIndex> new_rshape;
  vector<TIndex> new_oshape;
  int nDim = broadcast::BroadcastShapeCompact(lshape, rshape, oshape,
      &new_lshape, &new_rshape, &new_oshape);
  EXPECT_EQ(2u, nDim);
  vector<TIndex> expected_lshape = {6, 1};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_lshape[i], new_lshape[i]); 
  }
  vector<TIndex> expected_rshape = {6, 2};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_rshape[i], new_rshape[i]); 
  }
  vector<TIndex> expected_oshape = {6, 2};  
  for (int i = 0; i < nDim; i++) {
    EXPECT_EQ(expected_oshape[i], new_oshape[i]); 
  }
}

TEST(TestBroadcastElementwise, calc_stride) {
  vector<TIndex> shape = {3, 2, 2, 1}; 
  Shape<4> stride = calc_stride<4>(shape);
  vector<TIndex> expected_stride = {4, 2, 1, 0};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(expected_stride[i], stride[i]);
  }
}

TEST(TestBroadcastElementwise, unravel) {
  int idx = 9;
  vector<TIndex> vec_shape = {3, 2, 2, 2};
  Shape<4> shape(vec_shape);
  Shape<4> ret;
  unravel(idx, shape, &ret);
  vector<TIndex> expected_ret = {1, 0, 0, 1};  
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(expected_ret[i], ret[i]);
  } 
}

} // namespace broadcast
} // namespace blaze

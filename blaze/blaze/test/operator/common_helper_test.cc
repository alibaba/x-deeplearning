/*
 * \file common_helper_test.h
 * \brief The common helper test
 */
#include "gtest/gtest.h"

#include <math.h>
#include <thread>

#include "blaze/operator/common_helper.h"

namespace blaze {

TEST(CommonHelper, GetSliceAxis) {
  OperatorDef op_def;
  auto* argument = op_def.add_arg();
  argument->set_name("axis");
  argument->set_i(1);
  ArgumentHelper argument_helper(op_def);
  EXPECT_EQ(1, CommonHelper::GetSliceAxis(&argument_helper));
}

TEST(CommonHelper, GetSliceAxis2) {
  OperatorDef op_def;
  auto* argument = op_def.add_arg();
  argument->set_name("axes");
  argument->add_ints(1);
  ArgumentHelper argument_helper(op_def);
  EXPECT_EQ(1, CommonHelper::GetSliceAxis(&argument_helper));
}

TEST(CommonHelper, GetSliceStart) {
  OperatorDef op_def;
  auto* argument = op_def.add_arg();
  argument->set_name("start");
  argument->set_i(2);
  ArgumentHelper argument_helper(op_def);
  EXPECT_EQ(2, CommonHelper::GetSliceStart(&argument_helper));
}

TEST(CommonHelper, GetSliceStart2) {
  OperatorDef op_def;
  auto* argument = op_def.add_arg();
  argument->set_name("starts");
  argument->add_ints(2);
  ArgumentHelper argument_helper(op_def);
  EXPECT_EQ(2, CommonHelper::GetSliceStart(&argument_helper));
}

TEST(AttrMap, SetAttr) {
  AttrMap attr_map;
  attr_map.SetAttr("key", 1.0);
  EXPECT_FLOAT_EQ(attr_map.GetAttr("key", 2.0), 1.0);

  attr_map.SetAttr("key", 2);
  EXPECT_EQ(attr_map.GetAttr("key", 3), 2);
}

TEST(NElemFromDim, ElemFromDim) {
  TensorShape tensor_shape;
  tensor_shape.add_dims(2);
  tensor_shape.add_dims(3);
  tensor_shape.add_dims(5);
  size_t num = NElemFromDim(tensor_shape);
  EXPECT_EQ(num, 30);
}

TEST(GetIndicatorLevel, IndicatorLevel) {
  std::string indicator_name = "indicator.0";
  int level = GetIndicatorLevel(indicator_name);
  EXPECT_EQ(level, 0);
}

}  // namespace blaze
 

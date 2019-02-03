/*
 * \file proto_helper_test.cc
 * \brief The proto helper test module
 */
#include "gtest/gtest.h"

#include "blaze/common/types.h"
#include "blaze/common/proto_helper.h"

namespace blaze {

TEST(TestArgumentHelper, NetDefArgument) {
  NetDef net_def;
  Argument* argument = net_def.add_arg();
  argument->set_name("key");
  argument->set_i(12);
  EXPECT_TRUE(ArgumentHelper::HasArgument(net_def, "key"));
  EXPECT_EQ(ArgumentHelper::GetSingleArgument(net_def, "key", 1), 12);
  bool ret = ArgumentHelper::HasSingleArgumentOfType<NetDef, int>(net_def, "key");
  EXPECT_TRUE(ret);
}

TEST(TestArgumentHelper, SetArgument) {
  OperatorDef op_def;
  ArgumentHelper::SetSingleArgument<float>(op_def, "value", 1.0);
  EXPECT_EQ(1.0, ArgumentHelper::GetSingleArgument(op_def, "value", 2.0));

  ArgumentHelper::SetSingleArgument<float16>(op_def, "value", 1.0);
  EXPECT_EQ(1.0, ArgumentHelper::GetSingleArgument(op_def, "value", 2.0));
}

TEST(TestArgumentHelper, SetRepeatedArgument) {
  OperatorDef op_def;
  ArgumentHelper::SetRepeatedArgument<float>(op_def, "value", { 1.0, 2.0 });
  std::vector<float> value = ArgumentHelper::GetRepeatedArgument<OperatorDef, float>(op_def, "value");
  EXPECT_EQ(value.size(), 2);
  EXPECT_EQ(value[0], 1.0);
  EXPECT_EQ(value[1], 2.0);
}

TEST(TestArgumentHelper, ClearArgument) {
  OperatorDef op_def;
  ArgumentHelper::SetRepeatedArgument<float>(op_def, "value", { 1.0, 2.0 });
  ArgumentHelper::ClearArgument(op_def);
  EXPECT_EQ(op_def.arg_size(), 0);
}

TEST(TestLoadNetDefFromBinaryFile, LoadNetDefFromBinaryFile) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromBinaryFile("../benchmark/bench_data/graph/dnn.blaze.dat", &net_def);
  EXPECT_TRUE(ret);

  ret = NetDefHelper::SaveNetDefToBinaryFile("./net_def.sv.log", &net_def);
  EXPECT_TRUE(ret);

  ret = NetDefHelper::SaveNetDefToTextFile("./net_def.sv.txt.log", &net_def);
  EXPECT_TRUE(ret);
}

}  // namespace blaze



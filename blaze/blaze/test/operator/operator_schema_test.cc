/*
 * \file operator_schema_test.h
 * \brief The operator schema test
 */
#include "gtest/gtest.h"

#include <math.h>
#include <thread>

#include "blaze/common/proto_helper.h"
#include "blaze/operator/operator_schema.h"

namespace blaze {

TEST(TestOpSchema, OpSchame) {
  {
  OpSchema op_schema;
  EXPECT_STREQ(op_schema.file().c_str(), "unknown");
  EXPECT_EQ(op_schema.line(), 0);
  }
  
  {
  OpSchema op_schema("a.cpp", 1);
  EXPECT_STREQ(op_schema.file().c_str(), "a.cpp");
  EXPECT_EQ(op_schema.line(), 1);
  }
}

TEST(TestOpSchema, NumInputs) {
  OpSchema op_schema;
  op_schema.NumInputs(1);
  EXPECT_EQ(op_schema.min_input(), 1);
  EXPECT_EQ(op_schema.max_input(), 1);

  op_schema.NumInputs(1, 2);
  EXPECT_EQ(op_schema.min_input(), 1);
  EXPECT_EQ(op_schema.max_input(), 2);
}

TEST(TestOpSchema, NumInputs2) {
  OpSchema op_schema;
  std::set<int> allowed_input_nums;
  allowed_input_nums.insert(1);
  allowed_input_nums.insert(2);
  op_schema.NumInputs(allowed_input_nums);
  EXPECT_TRUE(op_schema.num_inputs_allowed(1));
  EXPECT_TRUE(op_schema.num_inputs_allowed(2));
  EXPECT_FALSE(op_schema.num_inputs_allowed(3));
}

TEST(TestOpSchema, NumOutputs) {
  OpSchema op_schema;
  op_schema.NumOutputs(2);
  EXPECT_EQ(op_schema.min_output(), 2);
  EXPECT_EQ(op_schema.max_output(), 2);

  op_schema.NumOutputs(2, 3);
  EXPECT_EQ(op_schema.min_output(), 2);
  EXPECT_EQ(op_schema.max_output(), 3);
}

TEST(TestOpSchema, NumOutputs2) {
  OpSchema op_schema;
  std::set<int> allowed_output_nums;
  allowed_output_nums.insert(1);
  allowed_output_nums.insert(2);
  op_schema.NumOutputs(allowed_output_nums);
  EXPECT_TRUE(op_schema.num_outputs_allowed(1));
  EXPECT_TRUE(op_schema.num_outputs_allowed(2));
  EXPECT_FALSE(op_schema.num_outputs_allowed(3));
}

TEST(TestOpSchema, NumInputsOutputs) {
  OpSchema op_schema;
  op_schema.NumInputsOutputs([](int a, int b) { return true; });
  EXPECT_TRUE(op_schema.num_inputs_outputs_allowed(1, 2));
}

TEST(TestOpSchema, AllowInplace) {
  OpSchema op_schema;
  EXPECT_FALSE(op_schema.allow_inplace(0, 0));
  op_schema.AllowInplace({ { 0, 0 } });
  EXPECT_TRUE(op_schema.allow_inplace(0, 0));
}

TEST(TestOpSchema, AllowOneToOneInplace) {
  OpSchema op_schema;
  op_schema.AllowOneToOneInplace();
  EXPECT_TRUE(op_schema.allow_inplace(0, 0));
  EXPECT_FALSE(op_schema.allow_inplace(0, 1));
}

TEST(TestOpSchema, IndenticalType) {
  OpSchema op_schema;
  op_schema.IdenticalType();
  OperatorDef def;
  std::vector<DataType> data_type = { kFloat };
  auto out_data_type = op_schema.InferType(def, data_type);
  EXPECT_EQ(1, out_data_type.size());
  EXPECT_EQ(kFloat, out_data_type[0]);
}

TEST(TestOpSchema, ScalarType) {
  OpSchema op_schema;
  op_schema.ScalarType(kFloat);
  OperatorDef def;
  def.add_output("ds1");
  def.add_output("ds2");
  std::vector<DataType> ins;
  auto out_data_type = op_schema.InferType(def, ins);
  EXPECT_EQ(2, out_data_type.size());
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(kFloat, out_data_type[i]);
  }
}

TEST(TestOpSchema, IdenticalShapeOfInput) {
  OpSchema op_schema;
  op_schema.IdenticalShapeOfInput(0);
  OperatorDef def;
  def.add_output("ds1");
  def.add_output("ds2");
  std::vector<TensorShape> shapes;
  TensorShape shape;
  shape.add_dims(2);
  shape.add_dims(3);
  shapes.push_back(shape);

  auto out_shape = op_schema.InferShape(def, shapes);
  EXPECT_EQ(out_shape.size(), 2);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(2, out_shape[i].dims_size());
    EXPECT_EQ(2, out_shape[i].dims(0));
    EXPECT_EQ(3, out_shape[i].dims(1));
  }
}

TEST(TestOpSchema, IdenticalShape) {
   OpSchema op_schema;
  op_schema.IdenticalShape();
  OperatorDef def;
  def.add_output("ds1");
  def.add_output("ds2");
  std::vector<TensorShape> shapes;
  TensorShape shape;
  shape.add_dims(2);
  shape.add_dims(3);
  shapes.push_back(shape);

  auto out_shape = op_schema.InferShape(def, shapes);
  EXPECT_EQ(out_shape.size(), 1);
  for (size_t i = 0; i < 1; ++i) {
    EXPECT_EQ(2, out_shape[i].dims_size());
    EXPECT_EQ(2, out_shape[i].dims(0));
    EXPECT_EQ(3, out_shape[i].dims(1));
  }
}

TEST(TestOpSchema, SetAttr) {
  OpSchema op_schema;
  op_schema.SetAttr<bool>("is_elementwise", true);
  EXPECT_TRUE(op_schema.GetAttr<bool>("is_elementwise", false));
}

TEST(TestElementWiseCostInference, All) {
  OperatorDef def;

  std::vector<TensorShape> input_shape;
  TensorShape shape;
  shape.add_dims(2);
  shape.add_dims(3);
  input_shape.push_back(shape);
  std::vector<DataType> input_type;
  input_type.push_back(kFloat);

  std::vector<TensorShape> output_shape;
  shape.clear_dims();
  shape.add_dims(2);
  shape.add_dims(3);
  output_shape.push_back(shape);
  std::vector<DataType> output_type;
  output_type.push_back(kFloat);

  auto cost = ElementWiseCostInference<1>(def, input_shape, input_type, output_shape, output_type);
  EXPECT_EQ(6, cost.flops);
  EXPECT_EQ(24, cost.bytes_written);
  EXPECT_EQ(24, cost.bytes_read);
}

TEST(TestInferTensorShape, InferTensorShape) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/operator/shape_infer.conf", &net_def);
  EXPECT_TRUE(ret);

  auto tensor_shape = InferTensorShape(net_def);
  for (const auto& iter : tensor_shape) {
    LOG_INFO("name=%s", iter.first.c_str());
    for (const auto& dim : iter.second.dims()) {
      LOG_INFO(" dim=%d", dim);
    }
  }
  // process att_ncomm
  auto att_comm_shape = tensor_shape["att_ncomm"];
  EXPECT_EQ(2, att_comm_shape.dims_size());
  EXPECT_EQ(kL0BatchSize, att_comm_shape.dims(0));
  EXPECT_EQ(kUnkownDim, att_comm_shape.dims(1));

  // process beta
  auto beta_shape = tensor_shape["beta"];
  EXPECT_EQ(1, beta_shape.dims_size());
  EXPECT_EQ(3, beta_shape.dims(0));
}

}  // namespace blaze


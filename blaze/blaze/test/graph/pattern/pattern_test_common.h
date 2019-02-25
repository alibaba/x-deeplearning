/*
 * \file pattern_test_common.h
 * \brief The operator test common
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/common/proto_helper.h"
#include "blaze/operator/operator.h"
#include "blaze/graph/workspace.h"
#include "blaze/graph/net.h"
#include "blaze/graph/fusion_pattern.h"

namespace blaze {

void ArgumentEqual(const Argument& arg, const Argument& arg_expected) {
  EXPECT_STREQ(arg_expected.name().c_str(), arg.name().c_str());
  EXPECT_EQ(arg_expected.has_f(), arg.has_f());
  EXPECT_EQ(arg_expected.has_i(), arg.has_i());
  EXPECT_EQ(arg_expected.has_s(), arg.has_s());
  if (arg_expected.has_f()) {
    EXPECT_FLOAT_EQ(arg_expected.f(), arg.f());
  }
  if (arg_expected.has_i()) {
    EXPECT_EQ(arg_expected.i(), arg.i());
  }
  if (arg_expected.has_s()) {
    EXPECT_EQ(arg_expected.s(), arg.s());
  }
  EXPECT_EQ(arg_expected.floats_size(), arg.floats_size());
  for (size_t i = 0; i < arg_expected.floats_size(); ++i) {
    EXPECT_FLOAT_EQ(arg_expected.floats(i), arg.floats(i));
  }

  EXPECT_EQ(arg_expected.ints_size(), arg.ints_size());
  for (size_t i = 0; i < arg_expected.ints_size(); ++i) {
    EXPECT_EQ(arg_expected.ints(i), arg.ints(i));
  }

  EXPECT_EQ(arg_expected.strings_size(), arg.strings_size());
  for (size_t i = 0; i < arg_expected.strings_size(); ++i) {
    EXPECT_STREQ(arg_expected.strings(i).c_str(), arg.strings(i).c_str());
  }
}

void ValueInfoEqual(const ValueInfo& value_info, const ValueInfo& value_info_expected) {
  EXPECT_STREQ(value_info_expected.name().c_str(), value_info.name().c_str());
  EXPECT_EQ(value_info_expected.dtype(), value_info.dtype());
  EXPECT_STREQ(value_info_expected.doc_string().c_str(), value_info.doc_string().c_str());
}

void OperatorDefEqual(const OperatorDef& op_def, const OperatorDef& op_def_expected) {
  EXPECT_STREQ(op_def_expected.type().c_str(), op_def.type().c_str());
  EXPECT_STREQ(op_def_expected.name().c_str(), op_def.name().c_str());
  EXPECT_EQ(op_def_expected.input_size(), op_def.input_size());
  EXPECT_EQ(op_def_expected.output_size(), op_def.output_size());
  for (size_t i = 0; i < op_def_expected.input_size(); ++i) {
    EXPECT_STREQ(op_def_expected.input(i).c_str(), op_def.input(i).c_str());
  }
  for (size_t i = 0; i < op_def_expected.output_size(); ++i) {
    EXPECT_STREQ(op_def_expected.output(i).c_str(), op_def.output(i).c_str());
  }
  EXPECT_EQ(op_def_expected.arg_size(), op_def.arg_size());
  for (size_t i = 0; i < op_def_expected.arg_size(); ++i) {
    ArgumentEqual(op_def.arg(i), op_def_expected.arg(i));
  }
}

void NetDefEqual(const NetDef& net_def, const NetDef& net_def_expected) {
  // Step1: check name equals
  EXPECT_STREQ(net_def_expected.name().c_str(), net_def.name().c_str());

  // Step2: check op
  EXPECT_EQ(net_def_expected.op_size(), net_def.op_size());
  for (size_t i = 0; i < net_def_expected.op_size(); ++i) {
    OperatorDefEqual(net_def.op(i), net_def_expected.op(i));
  }

  // Step3: check argument
  EXPECT_EQ(net_def_expected.arg_size(), net_def.arg_size());
  for (size_t i = 0; i < net_def_expected.arg_size(); ++i) {
    ArgumentEqual(net_def.arg(i), net_def_expected.arg(i));
  }

  // Step4: check external inputs or outputs
  EXPECT_EQ(net_def_expected.external_input_size(), net_def.external_input_size());
  EXPECT_EQ(net_def_expected.external_output_size(), net_def.external_output_size());
  for (size_t i = 0; i < net_def_expected.external_input_size(); ++i) {
    ValueInfoEqual(net_def.external_input(i), net_def_expected.external_input(i));
  }
  for (size_t i = 0; i < net_def_expected.external_output_size(); ++i) {
    ValueInfoEqual(net_def.external_output(i), net_def_expected.external_output(i));
  }
}

// Now only support cpu mode.
void CheckPatternOutput(const char* net_conf_path, const char* expected_net_conf_path) {
  Workspace workspace;
  std::shared_ptr<Net> net = workspace.CreateNet(net_conf_path);
  NetDef net_def = net->net_def();

  // Start to pass
  net_def = FusionPatternPass(net_def);

  LOG_INFO("FusionPatternPass Result: %s", net_def.DebugString().c_str());

  NetDef expected_net_def;
  EXPECT_TRUE(NetDefHelper::LoadNetDefFromTextFile(expected_net_conf_path, &expected_net_def));
  NetDefEqual(net_def, expected_net_def);
}

}  // namespace blaze

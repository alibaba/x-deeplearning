/*
 * \file ulf_importer_test.cc
 * \brief The ulf importer test unit
 */

#include <vector>
#include <unordered_set>
#include <string>

#include "gtest/gtest.h"
#include "blaze/common/log.h"
#include "blaze/model_importer/ulf_importer.h"

using std::vector;
using std::string;
using std::unordered_set;

namespace blaze {

class TestUlfImporter: public ::testing::Test {
public: 
  TestUlfImporter() {
    const char* net_conf_path = "./utest_data/model_importer/ulf/gauss_cnxh_din_v2/net-parameter-conf";
    const char* net_param_path = "./utest_data/model_importer/ulf/gauss_cnxh_din_v2/dnn-model-dat";
    importer_.LoadModel(net_conf_path, net_param_path);
  } 
 
  void SetUp() { 
  }
 
  void TearDown() { 
  }
 
  ~TestUlfImporter()  {
  }

  ULFImporter importer_;
};

TEST_F(TestUlfImporter, CreateConstantFillNode) {
  const NetDef& net_def = importer_.net_def();
  int count = 0;
  vector<const OperatorDef*> constant_ops;
  for (int i = 0; i < net_def.op_size(); i++) {
    if (net_def.op(i).type() == "ConstantFill") {
      count++;
      constant_ops.push_back(&net_def.op(i));
    }
  }
  EXPECT_EQ(111, count);
  unordered_set<string> gru60_set;
  for (int i = 0; i < 4; ++i) {
    gru60_set.insert("gru60_" + std::to_string(i));
  }
  int find_cnt = 0;
  for (int i = 0; i < count; ++i) {
    find_cnt += gru60_set.find(constant_ops[i]->name()) != gru60_set.end() ? 1 : 0;
    gru60_set.erase(constant_ops[i]->name());
  }
  EXPECT_EQ(4, find_cnt);
}

TEST_F(TestUlfImporter, ExternalNodes) {
  const NetDef& net_def = importer_.net_def();
  // check external input
  EXPECT_EQ(3u, net_def.external_input_size());
  EXPECT_STREQ("att_comm", net_def.external_input(0).name().c_str());
  EXPECT_STREQ("comm", net_def.external_input(1).name().c_str());
  EXPECT_STREQ("ncomm", net_def.external_input(2).name().c_str());

  // check external output
  EXPECT_EQ(1u, net_def.external_output_size());
  EXPECT_STREQ("output", net_def.external_output(0).name().c_str()); 
}

TEST_F(TestUlfImporter, ProcessBnLayer) {
  const NetDef& net_def = importer_.net_def();
  vector<const OperatorDef*> bns;
  for (int i = 0; i < net_def.op_size(); ++i) {
    if ("BatchNormalization" == net_def.op(i).type()) {
      bns.push_back(&net_def.op(i)); 
    } 
  }
  for (int i = 0; i < bns.size(); ++i) {
    EXPECT_EQ(5u, bns[i]->input_size());
    vector<string> bn_inputs(5u);
    for (int j = 0; j < 4; ++j) {
      bn_inputs[j + 1] = bns[i]->name() + "_" + std::to_string(j);  
    }
    std::swap(bn_inputs[1], bn_inputs[2]);
    for (int j = 1; j < bns[i]->input_size(); ++j) {
      EXPECT_STREQ(bn_inputs[j].c_str(), bns[i]->input(j).c_str()); 
    }   
  }   
}


} // namespace blaze

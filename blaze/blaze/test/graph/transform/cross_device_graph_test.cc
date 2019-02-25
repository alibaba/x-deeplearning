/*
 * \file cross_device_graph_test.cc
 * \brief The cross device graph test unit
 */

#include <vector>

#include "gtest/gtest.h"
#include "blaze/graph/transform/cross_device_graph.h"
#include "blaze/common/proto_helper.h"

namespace blaze {

TEST(TestCrossDeviceGraph, Total) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/transform/cross_device_graph.blaze", &net_def);
  ASSERT_TRUE(ret);
  CrossDeviceGraph graph(net_def);  

  // check the bridge node 
  std::vector<Node> bridge_nodes; 
  graph.BFS([&bridge_nodes](Node& node, void* arg) {
        if (node.op.type() == "Bridge") {
          bridge_nodes.push_back(node);
        }
        return false;
      }, nullptr);
  ASSERT_EQ(3u, bridge_nodes.size());
  const OperatorDef& op_def = bridge_nodes[0].op;  
  ASSERT_EQ(1u, op_def.input_size());
  EXPECT_STREQ("Slice-0-Output", op_def.input(0).c_str()); 
  EXPECT_STREQ("Slice-0-Output_bridge", op_def.output(0).c_str()); 
  EXPECT_STREQ("Bridge", op_def.type().c_str());
  EXPECT_EQ(1/*GPU*/, op_def.device_option().device_type());
  EXPECT_TRUE(op_def.device_option().is_pipe());
  EXPECT_EQ(1u, bridge_nodes[0].children.size()); 
  EXPECT_EQ(1u, bridge_nodes[0].children.begin()->second.size());
  EXPECT_STREQ("Slice-0-Output_bridge", bridge_nodes[0].children.begin()->second.at(0).c_str());
}

} // namespace

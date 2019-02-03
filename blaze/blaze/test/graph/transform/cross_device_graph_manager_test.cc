/*
 * \file cross_device_graph_manager_test.cc
 * \brief The cross device graph manager test unit
 */

#include <vector>

#include "gtest/gtest.h"
#include "blaze/graph/transform/cross_device_graph_manager.h"
#include "blaze/common/proto_helper.h"

namespace blaze {

TEST(TestCrossDeviceGraphManager, TestSplitToMultiParts) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/transform/cross_device_graph.blaze", &net_def);
  ASSERT_TRUE(ret);

  CrossDeviceGraphManager manager(net_def);
  std::vector<std::vector<int>> multi_parts; 
  CrossDeviceGraphManager::SplitToMultiParts(&net_def, &multi_parts); 
  EXPECT_EQ(3u, multi_parts.size());

  Graph graph(net_def);
  // first device is CPU
  {
    const std::vector<int>& parts = multi_parts[0];
    for (int i = 0; i < parts.size(); ++i) {
      EXPECT_EQ(0/*CPU*/, graph.device_option(graph.node(parts[i])).device_type());  
      EXPECT_FALSE(graph.device_option(graph.node(parts[i])).is_pipe());  
    }
  }
  // second device is pipe 
  {
    const std::vector<int>& parts = multi_parts[1];
    for (int i = 0; i < parts.size(); ++i) {
      EXPECT_EQ(1/*GPU*/, graph.device_option(graph.node(parts[i])).device_type());  
      EXPECT_TRUE(graph.device_option(graph.node(parts[i])).is_pipe());  
    }
  }
  // third device is GPU 
  {
    const std::vector<int>& parts = multi_parts[2];
    for (int i = 0; i < parts.size(); ++i) {
      EXPECT_EQ(1/*GPU*/, graph.device_option(graph.node(parts[i])).device_type());  
      EXPECT_FALSE(graph.device_option(graph.node(parts[i])).is_pipe());  
    }
  }
}

TEST(TestCrossDeviceGraphManager, TestTransform) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/transform/cross_device_graph.blaze", &net_def);
  ASSERT_TRUE(ret);
  
  CrossDeviceGraphManager manager(net_def);
  manager.Transform(); 
  const std::vector<NetDef>& net_defs = manager.GetNetDefs();
  ASSERT_EQ(3u, net_defs.size());

  // first device is CPU
  {
    EXPECT_EQ(0/*CPU*/, net_defs[0].device_option().device_type());
    EXPECT_FALSE(net_defs[0].device_option().is_pipe());  
    Graph graph(net_defs[0]);
    for (int i = 0; i < graph.size(); ++i) {
      EXPECT_EQ(0/*CPU*/, graph.device_option(graph.node(i)).device_type());  
      EXPECT_FALSE(graph.device_option(graph.node(i)).is_pipe());  
    }
  }
  // second device is pipe 
  {
    EXPECT_EQ(1/*GPU*/, net_defs[1].device_option().device_type());
    EXPECT_TRUE(net_defs[1].device_option().is_pipe());  
    Graph graph(net_defs[1]);
    for (int i = 0; i < graph.size(); ++i) {
      EXPECT_EQ(1/*GPU*/, graph.device_option(graph.node(i)).device_type());  
      EXPECT_TRUE(graph.device_option(graph.node(i)).is_pipe());  
    }
  }
  // third device is GPU 
  {
    EXPECT_EQ(1/*GPU*/, net_defs[2].device_option().device_type());
    EXPECT_FALSE(net_defs[2].device_option().is_pipe());  
    Graph graph(net_defs[2]);
    for (int i = 0; i < graph.size(); ++i) {
      EXPECT_EQ(1/*GPU*/, graph.device_option(graph.node(i)).device_type());  
      EXPECT_FALSE(graph.device_option(graph.node(i)).is_pipe());  
    }
  }
}

} // namespace

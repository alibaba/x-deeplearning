/*
 * \file graph_test.cc
 * \brief The graph test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/graph/graph.h"
#include "blaze/common/proto_helper.h"
#include "blaze/common/log.h"

namespace blaze {

bool Visit(Node& node, void* arg) {
  LOG_INFO("Visit node=%d", node.idx);
  return true;
}

TEST(TestGraph, TestAll) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/graph.blaze", &net_def);
  EXPECT_TRUE(ret);
  Graph graph(net_def);

  LOG_INFO("\n%s", graph.DebugStr().c_str());

  graph.BFS(Visit, NULL);
}

TEST(TestGraph, TestRenameInput) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/graph.blaze", &net_def);
  EXPECT_TRUE(ret);
  Graph graph(net_def);    
  // idx=9, gemm0, input: Concat-0-Output -> Concat-1-Output  
  int new_idx = graph.RenameInput(9, "Concat-0-Output", "Concat-1-Output");
  const Node& node = graph.node(new_idx);
  int idx;
  idx = node.GetParentIdx("Concat-0-Output");
  EXPECT_EQ(-1, idx);
  idx = node.GetParentIdx("Concat-1-Output");
  EXPECT_TRUE(idx >= 0);
  idx = node.GetParentIdx("gemm0-weight");
  EXPECT_TRUE(idx >= 0);
}

TEST(TestGraph, TestComplementNodes) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/small_graph.blaze", &net_def);
  EXPECT_TRUE(ret);
  Graph graph(net_def);
  std::vector<int> src_nodes = {1};
  std::vector<int> dst_nodes;
  ASSERT_TRUE(graph.ComplementNodes(src_nodes, &dst_nodes));
  EXPECT_EQ(2u, dst_nodes.size());
  std::vector<int> expected_nodes = {0, 2};
  for (int i = 0; i < dst_nodes.size(); ++i) {
    EXPECT_EQ(expected_nodes[i], dst_nodes[i]);
  }
}

TEST(TestGraph, TestMaxConnectedSearch) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/hybrid_graph.blaze", &net_def);
  EXPECT_TRUE(ret);
  Graph graph(net_def);
  std::vector<int> parts; 
  graph.MaxConnectedSearch([&graph, &parts](Node& node, void* arg) {
        if (parts.empty()) {
          parts.push_back(node.idx);
        } else {
          auto& first_device_option = graph.device_option(graph.node(parts[0])); 
          auto& cur_device_option = graph.device_option(graph.node(node.idx));
          if (graph.DeviceStr(first_device_option) != graph.DeviceStr(cur_device_option)) {
            return false;
          }
          parts.push_back(node.idx);
        } 
        return true; 
      }, nullptr);
  ASSERT_EQ(2u, parts.size());
  std::vector<int> expected_nodes = {1, 2};
  for (int i = 0; i < parts.size(); ++i) {
    EXPECT_EQ(expected_nodes[i], parts[i]);
  }
}

}  // namespace blaze

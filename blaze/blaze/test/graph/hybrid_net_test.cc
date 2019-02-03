/*
 * \file hybrid_net_test.cc
 * \brief The hybrid net test unit
 */

#include "gtest/gtest.h"
#include "blaze/graph/hybrid_net.h"

namespace blaze {

#ifdef USE_CUDA

TEST(TestHybridNet, TestConstructor) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/hybrid_net.blaze", &net_def);
  ASSERT_TRUE(ret);

  Workspace workspace;
  workspace.Init(net_def);
  std::shared_ptr<Net> net = workspace.CreateNet();
  HybridNet* hybrid_net = dynamic_cast<HybridNet*>(net.get()); 
  EXPECT_EQ(1u, hybrid_net->sub_nets_.size());
  EXPECT_EQ(0, hybrid_net->topo_next_net_.size());
  EXPECT_EQ(hybrid_net->external_input_.size(), hybrid_net->external_input_blob_.size());
  EXPECT_EQ(hybrid_net->external_output_.size(), hybrid_net->external_output_blob_.size());
}

TEST(TestHybridNet, TestNonBatching) {
  auto scheduler_manager = SchedulerManager<AsyncTask>::Instance();
  scheduler_manager->Destroy();
  SchedulerManager<AsyncTask>::Options options;
  options.enable_batching = false;
  options.num_threads_for_cpu = 2;
  options.num_threads_for_pipe = 2;
  options.num_threads_for_cuda = 2;
  ASSERT_TRUE(scheduler_manager->Init(options));

  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/hybrid_net.blaze", &net_def);
  ASSERT_TRUE(ret);

  Workspace workspace;
  workspace.Init(net_def);
  std::shared_ptr<Net> net = workspace.CreateNet();
  ASSERT_TRUE(net->Run());
}

TEST(TestHybridNet, TestBatching) {
  auto scheduler_manager = SchedulerManager<AsyncTask>::Instance();
  scheduler_manager->Destroy();
  SchedulerManager<AsyncTask>::Options options;
  options.enable_batching = true;
  options.max_batch_size = 1000;
  options.batch_timeout_micros = 10;
  options.num_threads_for_cpu = 2;
  options.num_threads_for_pipe = 2;
  options.num_threads_for_cuda = 2;
  ASSERT_TRUE(scheduler_manager->Init(options));

  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/hybrid_net.blaze", &net_def);
  ASSERT_TRUE(ret);

  Workspace workspace;
  workspace.Init(net_def);
  std::shared_ptr<Net> net = workspace.CreateNet();
  ASSERT_TRUE(net->Run());
}

TEST(TestHybridNet, TestGetTopoBlobName) {
  NetDef net_def;
  bool ret = NetDefHelper::LoadNetDefFromTextFile("./utest_data/graph/hybrid_net.blaze", &net_def);
  ASSERT_TRUE(ret);

  Workspace workspace;
  workspace.Init(net_def);
  std::shared_ptr<Net> net = workspace.CreateNet();
  auto blob_names = net->GetTopoBlobName();
  ASSERT_EQ(7u, blob_names.size());
  std::vector<std::string> expected_blob_names = {
    "ATT-COMM", "Slice-0-Output", "COMM", "Slice-1-Output", "Concat-0-Output",
    "gemm0-weight", "output"
  };
  for (int i = 0; i < blob_names.size(); ++i) {
    EXPECT_STREQ(expected_blob_names[i].c_str(), blob_names[i].c_str()); 
  }  
}

#endif

} // namespace blaze

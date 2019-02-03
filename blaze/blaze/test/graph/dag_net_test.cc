/*
 * \file simple_net_test.cc
 * \brief The simple net test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/graph/workspace.h"
#include "blaze/graph/net.h"
#include "blaze/common/log.h"
#include "blaze/common/timer.h"

namespace blaze {

#ifdef USE_CUDA
NetDef GetNetDef() {
  NetDef net_def;

  net_def.set_run_mode("dag");
  net_def.mutable_device_option()->set_device_type(kCUDA);
  
  OperatorDef* op = nullptr;
  Argument* arg = nullptr;

  // Layer 1: Slice layer
  op = net_def.add_op();
  op->set_type("Slice");
  op->set_name("Slice-1");
  op->add_input("ATT-COMM-1");
  op->add_output("Slice-1-Output");
  
  arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(1);
  
  arg = op->add_arg();
  arg->set_name("start");
  arg->set_i(0 * 1UL);
  
  arg = op->add_arg();
  arg->set_name("end");
  arg->set_i(6 * 1UL);

  // Layer 2: Slice layer
  op = net_def.add_op();
  op->set_type("Slice");
  op->set_name("Slice-2");
  op->add_input("ATT-COMM-2");
  op->add_output("Slice-2-Output");

  arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(1);
  
  arg = op->add_arg();
  arg->set_name("start");
  arg->set_i(6 * 1UL);
  
  arg = op->add_arg();
  arg->set_name("end");
  arg->set_i(12 * 1UL);

  // Layer 3: Slice layer
  op = net_def.add_op();
  op->set_type("Slice");
  op->set_name("Slice-3");
  op->add_input("ATT-COMM-3");
  op->add_output("Slice-3-Output");

  arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(1);
  
  arg = op->add_arg();
  arg->set_name("start");
  arg->set_i(4 * 1UL);
  
  arg = op->add_arg();
  arg->set_name("end");
  arg->set_i(10 * 1UL);

  // Layer 4: Slice layer
  op = net_def.add_op();
  op->set_type("Slice");
  op->set_name("Slice-4");
  op->add_input("ATT-COMM-4");
  op->add_output("Slice-4-Output");

  arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(1);
  
  arg = op->add_arg();
  arg->set_name("start");
  arg->set_i(6 * 1UL);
  
  arg = op->add_arg();
  arg->set_name("end");
  arg->set_i(12 * 1UL);

  // Layer 3: Concat layer
  op = net_def.add_op();
  op->set_type("Concat");
  op->set_name("Concat-1");
  op->add_input("Slice-1-Output");
  op->add_input("Slice-2-Output");
  op->add_input("Slice-3-Output");
  op->add_input("Slice-4-Output");
  op->add_output("Concat-1-Output");

  arg = op->add_arg();
  arg->set_name("axis");
  arg->set_i(1); 

  auto input = net_def.add_external_input();
  input->set_name("ATT-COMM-1");
  input->set_dtype(kFloat);

  auto input2 = net_def.add_external_input();
  input2->set_name("ATT-COMM-2");
  input2->set_dtype(kFloat);

  auto input3 = net_def.add_external_input();
  input3->set_name("ATT-COMM-3");
  input3->set_dtype(kFloat);

  auto input4 = net_def.add_external_input();
  input4->set_name("ATT-COMM-4");
  input4->set_dtype(kFloat);

  auto output = net_def.add_external_output();
  output->set_name("Concat-1-Output");
  output->set_dtype(kFloat);

  return net_def;
}

TEST(TestGraph, TestAll) {
  NetDef net_def = GetNetDef();
  
  Workspace workspace;
  DeviceOption device_option;
  device_option.set_device_type(kCUDA);

  workspace.Init(net_def);
  std::shared_ptr<Net> net = workspace.CreateNet();
  LOG_INFO("\n%s", net->DebugStr().c_str());

  workspace.CreateBlob("ATT-COMM-1", device_option)->Reshape({ 400UL, 100 * 1L, 32UL});
  workspace.CreateBlob("ATT-COMM-2", device_option)->Reshape({ 400UL, 100 * 1L, 32UL});
  workspace.CreateBlob("ATT-COMM-3", device_option)->Reshape({ 400UL, 100 * 1L, 32UL});
  workspace.CreateBlob("ATT-COMM-4", device_option)->Reshape({ 400UL, 100 * 1L, 32UL});
  workspace.CreateBlob("Slice-1-Output", device_option);
  workspace.CreateBlob("Slice-2-Output", device_option);
  workspace.CreateBlob("Slice-3-Output", device_option);
  workspace.CreateBlob("Slice-4-Output", device_option);
  workspace.CreateBlob("Concat-1-Output", device_option);

  net->RegisterObservers();
  const int kLoop = 100;
  Timer timer;
  timer.Start();
  for (int i = 0; i < kLoop; ++i) {
    net->Run();
  }
  timer.Stop();
  LOG_INFO("Outer time=%f", timer.GetElapsedTime() / kLoop);
  std::unordered_map<std::string, std::string> dump_map;
  net->Dump(dump_map);
}
#endif

}  // namespace blaze

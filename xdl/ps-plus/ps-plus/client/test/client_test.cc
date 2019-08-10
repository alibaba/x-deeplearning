/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "gtest/gtest.h"
#include "ps-plus/client/raw_client.h"
#include "ps-plus/client/client.h"
#include "ps-plus/client/local_client.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/initializer.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/message/worker_state.h"
#include <thread>

using ps::Data;
using ps::WrapperData;
using ps::Status;
using ps::Tensor;
using ps::client::Client;
using ps::client::RawClient;
using ps::client::LocalClient;
using ps::client::ClientArgs;
using ps::client::ClientWrapper;
using ps::client::Partitioner;
using ps::client::PartitionerContext;
using ps::client::UdfData;
using ps::client::UdfChain;
using ps::VariableInfo;
using ps::WorkerState;
using ps::initializer::ConstantInitializer;
using ps::initializer::NoneInitializer;

namespace {

class MockData {
 public:
  MockData() {
  }
  MockData(int value, int dim) {
    value_ = value;
    dim_ = dim;
  }
  int GetValue() const {
    return value_;
  }
  int GetDim() const {
    return dim_;
  }
  void AddValue(int value) {
    value_ += value;
  }
  void AddDim(int dim) {
    dim_ += dim;
  }
  int value_;
  int dim_;
};

class MockClientWrapper : public ClientWrapper {
 public:
  MockClientWrapper() {};
  void ReturnAsync(Status st, std::function<void(const Status&)> done) {
    std::thread response([st, done](){
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      done(st);
    });
    response.detach();
  }
  void MockRemoteVariableInfo(const std::vector<VariableInfo>& input) {
    variable_info_.clear();
    for (auto& item : input) {
      variable_info_[item.name] = item;
    }
  }
  void MockRemoteUdf(const std::vector<unsigned long long>& input) {
    mok_udf_.clear();
    for (auto& item : input) {
      mok_udf_.push_back(item);
    }
  }
  void UpdateVariableVisitInfo(const std::string& name, int64_t id_num, const Callback& cb) {}
  void UpdateVariableInfo(const std::vector<VariableInfo>& input, 
                          std::vector<VariableInfo>* output, 
                          const Callback& cb) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& item : input) {
      auto iter = variable_info_.find(item.name);
      if (iter == variable_info_.end()) {
        variable_info_[item.name] = item;
      }
    }

    for (auto& item : variable_info_) {
      (*output).push_back(item.second);
    }
    ReturnAsync(Status::Ok(), cb);
  }
  void Process(const std::string& var_name,
               size_t server_id, 
               size_t udf_id, 
               const std::vector<Data*>& input, 
               std::vector<Data*>* output, 
               const Callback& cb) {
    bool flag = false;
    for (size_t i = 0; i < mok_udf_.size(); ++i) {
      if (udf_id == mok_udf_[i]) {
        flag = true;
      }
    }

    if (!flag) {
      ReturnAsync(Status::UdfNotRegistered("Udf Not Registered."), cb);
      return;
    }

    // push type
    if (udf_id != mok_udf_[1]) {
      ReturnAsync(Status::Ok(), cb);
      return;
    }

    // pull type
    if (udf_id == mok_udf_[1]) {
      VariableInfo info = variable_info_[var_name];
      for (auto& item : info.parts) {
        if (server_id == item.server) {
          WrapperData<MockData>* data = new WrapperData<MockData>(1, item.size);
          (*output).push_back(data);
        }
      }
      ReturnAsync(Status::Ok(), cb);
      return;
    }
  }
  void RegisterUdf(size_t server_id, const UdfChain& def, const Callback& cb) {
    std::lock_guard<std::mutex> lock(mu_);
    mok_udf_.push_back(def.hash());
    ReturnAsync(Status::Ok(), cb);
    return;
  }
  Status ConnectToCluster(const std::string& addr) {
    return Status::Ok();
  }
  void Save(const std::string& version, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  };
  void Restore(const std::string& version, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  };
  void TriggerStreamingModelDense(const std::string& stream_ver, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  }

  Status InitGlobalQueue(
      const std::string& name,
      const std::vector<std::string>& paths, 
      size_t epochs, 
      bool epoch_isolate = false) {
    return Status::Ok();
  }
  Status GetNextFile(
      const std::string& name,
      size_t worker_id, 
      std::string* path, 
      size_t* begin, 
      size_t* epoch) {
    return Status::Ok();
  }
  Status ReportWorkerState(
      const std::string& name,
      size_t worker_id, 
      const std::vector<WorkerState>& worker_states) {
    return Status::Ok();
  }
  Status RestoreWorkerState(
      const std::string& name,
      size_t worker_id) {
    return Status::Ok();
  }
  void TriggerStreamingModelSparse(const std::string& stream_ver, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  };
  void TriggerStreamingModelHash(const std::string& stream_ver, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  };
  void AsynchronizeEnter(int id, int staleness, int worker_count, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  }    
  void SynchronizeEnter(int id, int worker_count, int64_t* token, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  }
  void SynchronizeLeave(int id, int64_t token, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  }
  void WorkerReportFinish(int id, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  }
  void GetWorkerFinishCount(int64_t* count, const Callback& cb) {
    *count = 0;
    ReturnAsync(Status::Ok(), cb);
    return;
  }  
  void WorkerBarrier(int id, int worker_count, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  }
  void WorkerBarrierV2(int barrier_id, int task_id, int task_num, int token, const Callback& cb) {
    ReturnAsync(Status::Ok(), cb);
    return;
  }
  virtual void ModelServerForward(int server_type, int server_id, const Tensor& ids, std::unique_ptr<Tensor>* rst, const Callback& cb) {}
  virtual void ModelServerBackward(int server_type, int server_id, const Tensor& ids, const Tensor& grads, const Callback& cb) {}
  virtual int ServerSize(int id) { return 0; }
  virtual int ServerTypeSize() { return 0; }
  
  // mok remote info on scheduler
  std::mutex mu_;
  std::unordered_map<std::string, VariableInfo> variable_info_; 
  std::vector<unsigned long long> mok_udf_;
};

class MockPartitioner : public Partitioner {
 public:
  Status Split(PartitionerContext* ctx, Data* src, std::vector<Data*>* dst) {
    VariableInfo* info = ctx->GetVariableInfo();
    int server = info->parts.size();
    MockData data = dynamic_cast<WrapperData<MockData>*>(src)->Internal();
    for(int i = 0; i < server; ++i) {
      WrapperData<MockData>* dst_data = new WrapperData<MockData>(data.GetValue(), info->parts[i].size);
      (*dst).push_back(dst_data);
    }
    return Status::Ok();
  }
  Status Combine(PartitionerContext* ctx, Data* src, size_t id, std::unique_ptr<Data>* dst) {
    std::lock_guard<std::mutex> lock(mu_);
    MockData data = dynamic_cast<WrapperData<MockData>*>(src)->Internal();
    if ((*dst) == nullptr) {
      int dim = data.GetDim();
      int value = data.GetValue();
      WrapperData<MockData>* dst_data = new WrapperData<MockData>(value, dim);
      (*dst).reset(dst_data);
    } else {
      MockData& dst_data = dynamic_cast<WrapperData<MockData>*>((*dst).get())->Internal();
      dst_data.AddValue(data.GetValue());
      dst_data.AddDim(data.GetDim());
    }
    return Status::Ok();
  }
  std::mutex mu_;
};

void MockArgument(std::vector<VariableInfo>& remote_info,
                 std::vector<unsigned long long>& remote_udf,
                 ClientArgs& args,
                 std::vector<Partitioner*>& splitter,
                 std::vector<Partitioner*>& splitter_error,
                 std::vector<Partitioner*>& combiner,
                 std::vector<Partitioner*>& combiner_error,
                 std::vector<Data*>& datas,
                 std::vector<std::unique_ptr<Data>>& results) {
  // Mock remote variable info and tensor placement on server
  // total server: 1, 2, 3, 4, 5
  VariableInfo info2 = {VariableInfo::kIndex, "var2", {{3, 3}, {4, 4}}};
  VariableInfo info3 = {VariableInfo::kIndex, "var3", {{1, 1}, {5, 5}}};  

  remote_info.push_back(info2);
  remote_info.push_back(info3);

  // Mock UdfChain
  UdfData udf_data1 = UdfData(1);
  UdfChain udf1 = UdfChain(udf_data1);

  UdfData udf_data2 = UdfData(2);
  UdfChain udf2 = UdfChain(udf_data2);

  remote_udf.push_back(udf1.hash());
  remote_udf.push_back(udf2.hash());  

  // Mock ClientArgs
  MockClientWrapper* mok_client_wrapper = new MockClientWrapper();
  mok_client_wrapper->MockRemoteVariableInfo(remote_info);
  mok_client_wrapper->MockRemoteUdf(remote_udf);

  args.scheduler_addr = "";
  args.client_wrapper_creator = [mok_client_wrapper](){ return mok_client_wrapper; };
  
  // Mock datas
  WrapperData<MockData>* data1 = new WrapperData<MockData>(1, 3);
  WrapperData<MockData>* data2 = new WrapperData<MockData>(2, 3);

  datas.push_back(data1);
  datas.push_back(data2);

  // Mock Partitioner
  for (int i = 0; i < 2; ++i) {
    splitter.push_back(new MockPartitioner());
  }
  for (int i = 0; i < 3; ++i) {
    splitter_error.push_back(new MockPartitioner());
  }
  
  for (int i = 0; i < 1; ++i) {
    combiner.push_back(new MockPartitioner());
  }

  for (int i = 0; i < 2; ++i) {
    combiner_error.push_back(new MockPartitioner());
  }

  results.clear();
}

}

template<typename T>
void DeleteItems(std::vector<T*>& arg) {
  for (auto item : arg) {
    delete item;
  }
}

template<typename T, typename... Args>
void DeleteItems(std::vector<T*>& head, Args&... rest) {
  for (auto item : head) {
    delete item;
  }
  DeleteItems(rest...);
}

TEST(ClientTest, ProcessVariableInfoNotFoundTest) {
  std::vector<VariableInfo> remote_info;
  std::vector<unsigned long long> remote_udf;
  ClientArgs args;
  std::vector<Partitioner*> splitter;
  std::vector<Partitioner*> splitter_error;
  std::vector<Partitioner*> combiner;
  std::vector<Partitioner*> combiner_error;
  std::vector<Data*> datas;
  std::vector<std::unique_ptr<Data>> results;

  MockArgument(remote_info, remote_udf, args, splitter, splitter_error, combiner, combiner_error, datas, results);
  Client* client = new Client(new RawClient(args));
  client->Init();

  UdfData udf_data1 = UdfData(1);
  UdfChain udf1 = UdfChain(udf_data1);

  std::promise<Status> st_promise;
  client->Process(udf1, "var4", datas, splitter, {}, &results, [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();

  EXPECT_EQ(Status::kArgumentError, st.Code());  

  DeleteItems(splitter_error, combiner, combiner_error);
  delete client;
}

TEST(ClientTest, ProcessSplitterWrongSizeTest) {
  std::vector<VariableInfo> remote_info;
  std::vector<unsigned long long> remote_udf;
  ClientArgs args;
  std::vector<Partitioner*> splitter;
  std::vector<Partitioner*> splitter_error;
  std::vector<Partitioner*> combiner;
  std::vector<Partitioner*> combiner_error;
  std::vector<Data*> datas;
  std::vector<std::unique_ptr<Data>> results;

  MockArgument(remote_info, remote_udf, args, splitter, splitter_error, combiner, combiner_error, datas, results);
  Client* client = new Client(new RawClient(args));
  client->Init();

  UdfData udf_data1 = UdfData(1);
  UdfChain udf1 = UdfChain(udf_data1);
  VariableInfo info1 = {VariableInfo::kIndex, "var1", {{1, 1}, {2, 2}}};
  client->RegisterVariable("var1", info1);

  std::promise<Status> st_promise;
  client->Process(udf1, "var1", datas, splitter_error, {}, &results, [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();

  EXPECT_EQ(Status::kArgumentError, st.Code());  
  EXPECT_EQ("Splitter has the wrong size.", st.Msg());

  DeleteItems(splitter, combiner, combiner_error);
  delete client;
}

TEST(ClientTest, ProcessCombinerWrongSizeTest) {
  std::vector<VariableInfo> remote_info;
  std::vector<unsigned long long> remote_udf;
  ClientArgs args;
  std::vector<Partitioner*> splitter;
  std::vector<Partitioner*> splitter_error;
  std::vector<Partitioner*> combiner;
  std::vector<Partitioner*> combiner_error;
  std::vector<Data*> datas;
  std::vector<std::unique_ptr<Data>> results;

  MockArgument(remote_info, remote_udf, args, splitter, splitter_error, combiner, combiner_error, datas, results);
  Client* client = new Client(new RawClient(args));
  client->Init();

  UdfData udf_data2 = UdfData(2);
  UdfChain udf2 = UdfChain(udf_data2);
  VariableInfo info1 = {VariableInfo::kIndex, "var1", {{1, 1}, {2, 2}}};
  client->RegisterVariable("var1", info1);
  
  std::promise<Status> st_promise;
  client->Process(udf2, "var1", {}, {}, combiner_error, &results, [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();

  EXPECT_EQ(Status::kArgumentError, st.Code());  
  EXPECT_EQ("Combiner Size Error", st.Msg());

  DeleteItems(splitter, splitter_error, combiner);
  delete client;
}

TEST(ClientTest, ProcessNormalPushTest) {
  std::vector<VariableInfo> remote_info;
  std::vector<unsigned long long> remote_udf;
  ClientArgs args;
  std::vector<Partitioner*> splitter;
  std::vector<Partitioner*> splitter_error;
  std::vector<Partitioner*> combiner;
  std::vector<Partitioner*> combiner_error;
  std::vector<Data*> datas;
  std::vector<std::unique_ptr<Data>> results;

  MockArgument(remote_info, remote_udf, args, splitter, splitter_error, combiner, combiner_error, datas, results);
  Client* client = new Client(new RawClient(args));
  client->Init();

  UdfData udf_data1 = UdfData(1);
  UdfChain udf1 = UdfChain(udf_data1);
  VariableInfo info1 = {VariableInfo::kIndex, "var1", {{1, 1}, {2, 2}}};
  client->RegisterVariable("var1", info1);
  
  std::promise<Status> st_promise;
  client->Process(udf1, "var1", datas, splitter, {}, &results, [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();

  EXPECT_EQ(Status::Ok(), st);  

  DeleteItems(splitter_error, combiner, combiner_error);
  delete client;
}

TEST(ClientTest, ProcessNormalPushResendTest) {
  std::vector<VariableInfo> remote_info;
  std::vector<unsigned long long> remote_udf;
  ClientArgs args;
  std::vector<Partitioner*> splitter;
  std::vector<Partitioner*> splitter_error;
  std::vector<Partitioner*> combiner;
  std::vector<Partitioner*> combiner_error;
  std::vector<Data*> datas;
  std::vector<std::unique_ptr<Data>> results;

  MockArgument(remote_info, remote_udf, args, splitter, splitter_error, combiner, combiner_error, datas, results);
  Client* client = new Client(new RawClient(args));
  client->Init();

  UdfData udf_data3 = UdfData(3);
  UdfChain udf3 = UdfChain(udf_data3);
  VariableInfo info1 = {VariableInfo::kIndex, "var1", {{1, 1}, {2, 2}}};
  client->RegisterVariable("var1", info1);
  
  std::promise<Status> st_promise;
  client->Process(udf3, "var1", datas, splitter, {}, &results, [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();

  EXPECT_EQ(Status::Ok(), st);  

  DeleteItems(splitter_error, combiner, combiner_error);
  delete client;
}

TEST(ClientTest, ProcessNormalPullTest) {
  std::vector<VariableInfo> remote_info;
  std::vector<unsigned long long> remote_udf;
  ClientArgs args;
  std::vector<Partitioner*> splitter;
  std::vector<Partitioner*> splitter_error;
  std::vector<Partitioner*> combiner;
  std::vector<Partitioner*> combiner_error;
  std::vector<Data*> datas;
  std::vector<std::unique_ptr<Data>> results;

  MockArgument(remote_info, remote_udf, args, splitter, splitter_error, combiner, combiner_error, datas, results);
  Client* client = new Client(new RawClient(args));
  client->Init();

  UdfData udf_data2 = UdfData(2);
  UdfChain udf2 = UdfChain(udf_data2);
  VariableInfo info1 = {VariableInfo::kIndex, "var1", {{1, 1}, {2, 2}}};
  client->RegisterVariable("var1", info1);
  
  std::promise<Status> st_promise;
  client->Process(udf2, "var1", {}, {}, combiner, &results, [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();

  EXPECT_EQ(Status::Ok(), st);
  MockData result_data = dynamic_cast<WrapperData<MockData>*>(results[0].get())->Internal();
  EXPECT_EQ(2, result_data.GetValue());
  EXPECT_EQ(3, result_data.GetDim());

  DeleteItems(splitter, splitter_error, combiner_error);
  delete client;
}

TEST(ClientTest, OtherTest) {
  std::vector<VariableInfo> remote_info;
  std::vector<unsigned long long> remote_udf;
  ClientArgs args;
  std::vector<Partitioner*> splitter;
  std::vector<Partitioner*> splitter_error;
  std::vector<Partitioner*> combiner;
  std::vector<Partitioner*> combiner_error;
  std::vector<Data*> datas;
  std::vector<std::unique_ptr<Data>> results;

  MockArgument(remote_info, remote_udf, args, splitter, splitter_error, combiner, combiner_error, datas, results);
  Client* client = new Client(new RawClient(args));
  client->Init();

  std::promise<Status> st_promise;
  client->Save("version_test", [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise1;
  client->Restore("version_test", [&st_promise1](Status st){
    st_promise1.set_value(st);
  });
  st = st_promise1.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::string ver("inc-test99");
  std::promise<Status> st_promise2;
  client->TriggerStreamingModelDense(ver, [&st_promise2](Status st){
    st_promise2.set_value(st);
  });
  st = st_promise2.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise3;
  client->TriggerStreamingModelSparse(ver, [&st_promise3](Status st){
    st_promise3.set_value(st);
  });
  st = st_promise3.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise4;
  client->TriggerStreamingModelHash(ver, [&st_promise4](Status st){
    st_promise4.set_value(st);
  });
  st = st_promise4.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise5;
  client->AsynchronizeEnter(0, 1, 2, [&st_promise5](Status st){
    st_promise5.set_value(st);
  });
  st = st_promise5.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise6;
  client->SynchronizeEnter(0, 2, [&st_promise6](Status st){
    st_promise6.set_value(st);
  });
  st = st_promise6.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise7;
  client->SynchronizeLeave(0, [&st_promise7](Status st){
    st_promise7.set_value(st);
  });
  st = st_promise7.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise8;
  client->WorkerReportFinish(0, [&st_promise8](Status st){
    st_promise8.set_value(st);
  });
  st = st_promise8.get_future().get();
  EXPECT_EQ(Status::Ok(), st);  
}

TEST(ClientTest, OtherRemoteTest) {
  std::vector<VariableInfo> remote_info;
  std::vector<unsigned long long> remote_udf;
  ClientArgs args;
  std::vector<Partitioner*> splitter;
  std::vector<Partitioner*> splitter_error;
  std::vector<Partitioner*> combiner;
  std::vector<Partitioner*> combiner_error;
  std::vector<Data*> datas;
  std::vector<std::unique_ptr<Data>> results;

  MockArgument(remote_info, remote_udf, args, splitter, splitter_error, combiner, combiner_error, datas, results);
  Client* client = new Client(new RawClient(args));
  client->Init();
  
  std::promise<Status> st_promise0;
  client->SynchronizeEnter(0, 2, [&st_promise0](Status st){
    st_promise0.set_value(st);
  });
  Status st = st_promise0.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise1;
  client->IndexInitializer("var4", new NoneInitializer(), [&st_promise1](Status st){
    st_promise1.set_value(st);
  });
  st = st_promise1.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::promise<Status> st_promise2;
  Tensor tensor;
  client->IdentityInitializer("var4", tensor, [&st_promise2](Status st){
    st_promise2.set_value(st);
  });
  st = st_promise2.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::promise<Status> st_promise3;
  client->HashInitializer("var4", new NoneInitializer(), [&st_promise3](Status st){
    st_promise3.set_value(st);
  });
  st = st_promise3.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::promise<Status> st_promise4;
  bool isbool;
  client->IsInitialized("var4", &isbool, [&st_promise4](Status st){
    st_promise4.set_value(st);
  });
  st = st_promise4.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::promise<Status> st_promise5;
  client->DensePull("var4", &tensor, [&st_promise5](Status st){
    st_promise5.set_value(st);
  });
  st = st_promise5.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::promise<Status> st_promise6;
  client->DensePush("var4", "updater", datas, [&st_promise6](Status st){
    st_promise6.set_value(st);
  });
  st = st_promise6.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::promise<Status> st_promise7;
  Tensor id;
  client->SparsePull("var4", id, &tensor, [&st_promise7](Status st){
    st_promise7.set_value(st);
  });
  st = st_promise7.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::vector<Data*> datas1;
  std::promise<Status> st_promise8;
  client->SparsePush("var4", id, "update", datas1, [&st_promise8](Status st){
    st_promise8.set_value(st);
  });
  st = st_promise8.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::promise<Status> st_promise9;
  client->HashPull("var4", id, 1.0, &tensor, [&st_promise9](Status st){
    st_promise9.set_value(st);
  });
  st = st_promise9.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());

  std::vector<Data*> datas2;
  std::promise<Status> st_promise10;
  client->HashPush("var4", id, 0.0, false, "update", datas2, [&st_promise10](Status st){
    st_promise10.set_value(st);
  });
  st = st_promise10.get_future().get();
  EXPECT_EQ(Status::kArgumentError, st.Code());
}

TEST(LocalClientTest, LocalTest) {
  auto client = new LocalClient("./");
  client->Init();

  std::promise<Status> st_promise;
  client->Save("version_test", [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise1;
  client->Restore("version_test", [&st_promise1](Status st){
    st_promise1.set_value(st);
  });
  st = st_promise1.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  Tensor ids;
  std::promise<Status> st_promise2;
  client->ModelServerForward(0, ids, &ids, [&st_promise2](Status st){
    st_promise2.set_value(st);
  });
  st = st_promise2.get_future().get();
  EXPECT_NE(Status::Ok(), st);

  std::promise<Status> st_promise3;
  client->ModelServerBackward(1, ids, ids, [&st_promise3](Status st){
    st_promise3.set_value(st);
  });
  st = st_promise3.get_future().get();
  EXPECT_NE(Status::Ok(), st);

  std::string ver("inc-test99");
  std::promise<Status> st_promise4;
  client->TriggerStreamingModelDense(ver, [&st_promise4](Status st) {
    st_promise4.set_value(st);
  });
  st = st_promise4.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise5;
  client->TriggerStreamingModelSparse(ver, [&st_promise5](Status st) {
    st_promise5.set_value(st);
  });
  st = st_promise5.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise6;
  client->TriggerStreamingModelHash(ver, [&st_promise6](Status st) {
    st_promise6.set_value(st);
  });
  st = st_promise6.get_future().get();
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise7;
  client->AsynchronizeEnter(0, 1, 2, [&st_promise7](Status st) {
    st_promise7.set_value(st);
  });
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise8;
  client->SynchronizeEnter(0, 1, [&st_promise8](Status st) {
    st_promise8.set_value(st);
  });
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise9;
  client->SynchronizeLeave(0, [&st_promise9](Status st) {
    st_promise9.set_value(st);
  });
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise10;
  client->WorkerReportFinish(1, [&st_promise10](Status st) {
    st_promise10.set_value(st);
  });
  EXPECT_EQ(Status::Ok(), st);

  std::promise<Status> st_promise11;

  client->HashInitializer("hello", new ConstantInitializer(0), [&st_promise11](Status st) {
    st_promise11.set_value(st);
  });
  st = st_promise11.get_future().get();
  EXPECT_NE(Status::Ok(), st);

  Tensor ids1;
  std::promise<Status> st_promise12;

  client->HashPull("hello", ids1, 1.0, &ids1, [&st_promise12](Status st) {
    st_promise12.set_value(st);
  });
  st = st_promise12.get_future().get();
  EXPECT_NE(Status::Ok(), st);

  Tensor ids2;
  std::promise<Status> st_promise13;

  client->HashPull("hello", ids2, 1.0, &ids2, [&st_promise13](Status st) {
    st_promise13.set_value(st);
  });
  st = st_promise13.get_future().get();
  EXPECT_NE(Status::Ok(), st);

}

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

#include "ps-plus/client/raw_client.h"
#include "ps-plus/client/process_context.h"
#include "ps-plus/client/merged_process_context.h"
#include "ps-plus/client/model_server_splitter.h"

#include "ps-plus/common/logging.h"

#include <iostream>
#include <sstream>
#include <sys/time.h>

#define RETURN_ASYNC(STATUS) do { cb(STATUS); return; } while (0)

#define CHECK_ASYNC(STATUS) do {                                                                                \
    Status st_ = STATUS;                                                                                        \
    if (!st_.IsOk()) {                                                                                          \
        st_.Msg() += "\nCHECKED BY [" #STATUS "] @ FILE[" __FILE__ "] LINE[" + std::to_string(__LINE__) + "]";  \
        RETURN_ASYNC(st_);                                                                                      \
    }                                                                                                           \
} while (0)


namespace ps {
namespace client {

RawClient::RawClient(const ClientArgs& args)
  : args_(args), init_variable_info_(false) {}

Status RawClient::Init() {
  {
    std::lock_guard<std::mutex> lock(variable_info_mutex_);
    init_variable_info_ = false;
  }
  client_wrapper_.reset(args_.client_wrapper_creator());
  return client_wrapper_->ConnectToCluster(args_.scheduler_addr);
}

void RawClient::Process(
  const UdfChain& udf, 
  const std::vector<std::string>& var_names,
  const std::vector<Data*>& datas,
  const std::vector<MergedPartitioner*>& splitter,
  const std::vector<MergedPartitioner*>& combiner,
  std::vector<std::vector<std::unique_ptr<Data>>>* results,
  const Callback& cb_internal) {

  MergedPartitionerContext* ctx = new MergedPartitionerContext;
  Callback cb = [this, ctx, splitter, combiner, datas, cb_internal](Status s){
    cb_internal(s);
    delete ctx;
    for (auto item : splitter) {
      delete item;
    }
    for (auto item : combiner) {
      delete item;
    }
    for (auto item : datas) {
      delete item;
    }
  };
  for (size_t i = 0; i < var_names.size(); ++i) {
    PartitionerContext* one_ctx = new PartitionerContext;
    VariableInfo info;
    CHECK_ASYNC(GetVariableInfo(var_names[i], &info));
    one_ctx->SetVariableInfo(info);
    ctx->AddContext(one_ctx);
  }
  const std::vector<VariableInfo::Part>& server_to_send = ctx->GetContext(0)->GetVariableInfo()->parts;
  size_t servers = server_to_send.size();

  if (splitter.size() != datas.size()) {
    RETURN_ASYNC(Status::ArgumentError("Splitter has the wrong size."));
  }

  std::vector<std::vector<Data*>> split_results;
  for (size_t i = 0; i < datas.size(); ++i) {
    CHECK_ASYNC(splitter[i]->Init(ctx, datas[i]));
  }
  for (size_t i = 0; i < datas.size(); ++i) {
    split_results.emplace_back();
    CHECK_ASYNC(splitter[i]->Split(ctx, datas[i], &split_results.back()));
  }

  std::vector<std::vector<Data*>> request(servers);  
  for (size_t i = 0; i < split_results.size(); ++i) {
    if (split_results[i].size() != servers) {
      RETURN_ASYNC(Status::ArgumentError("Splitter result size error"));
    }
    for (size_t j = 0; j < split_results[i].size(); ++j) {
      request[j].push_back(split_results[i][j]);
    }
  }
  
  MergedProcessContext* pctx = new MergedProcessContext(servers);
  (*results).clear();
  (*results).resize(combiner.size());
  for (auto& result_vec : (*results)) {
    result_vec.resize(var_names.size());
  }

  for (size_t i = 0; i < combiner.size(); ++i) {
    combiner[i]->CombineInit(ctx, &(*results)[i]);
  }

  for (size_t i = 0; i < servers; ++i) {
    size_t server_id = server_to_send[i].server;
    std::vector<Data*>* server_results = new std::vector<Data*>();
    Process("^hash_variable", server_id, udf, request[i], server_results, 
      pctx->CollectResults(combiner, ctx, server_results, results, i, cb));
  }
}

void RawClient::Process(
  const UdfChain& udf, 
  const std::string& var_name,
  const std::vector<Data*>& datas,
  const std::vector<Partitioner*>& splitter,
  const std::vector<Partitioner*>& combiner,
  std::vector<std::unique_ptr<Data>>* results,
  const Callback& cb_internal) {

  PartitionerContext* ctx = new PartitionerContext;
  Callback cb = [this, ctx, splitter, combiner, datas, cb_internal](Status s){
    cb_internal(s);
    delete ctx;
    for (auto item : splitter) {
      delete item;
    }
    for (auto item : combiner) {
      delete item;
    }
    for (auto item : datas) {
      delete item;
    }
  };

  VariableInfo info;
  CHECK_ASYNC(GetVariableInfo(var_name, &info));

  ctx->SetVariableInfo(info);
  const std::vector<VariableInfo::Part>& server_to_send = info.parts;
  size_t servers = server_to_send.size();

  if (splitter.size() != datas.size()) {
    RETURN_ASYNC(Status::ArgumentError("Splitter has the wrong size."));
  }

  std::vector<std::vector<Data*>> split_results;
  for (size_t i = 0; i < datas.size(); ++i) {
    CHECK_ASYNC(splitter[i]->Init(ctx, datas[i]));
  }
  for (size_t i = 0; i < datas.size(); ++i) {
    split_results.emplace_back();
    CHECK_ASYNC(splitter[i]->Split(ctx, datas[i], &split_results.back()));
  }

  std::vector<std::vector<Data*>> request(servers);  
  for (size_t i = 0; i < split_results.size(); ++i) {
    if (split_results[i].size() != servers) {
      RETURN_ASYNC(Status::ArgumentError("Splitter result size error"));
    }
    for (size_t j = 0; j < split_results[i].size(); ++j) {
      request[j].push_back(split_results[i][j]);
    }
  }
  
  ProcessContext* pctx = new ProcessContext(servers);
  (*results).clear();
  (*results).resize(combiner.size());

  for (size_t i = 0; i < combiner.size(); ++i) {
    combiner[i]->CombineInit(ctx, &(*results)[i]);
  }

  for (size_t i = 0; i < servers; ++i) {
    size_t server_id = server_to_send[i].server;
    std::vector<Data*>* server_results = new std::vector<Data*>();
    Process(var_name, server_id, udf, request[i], server_results, 
      pctx->CollectResults(combiner, ctx, server_results, results, i, cb));
  }
}

void RawClient::Save(const std::string& name, const Callback& cb) {
  client_wrapper_->Save(name, cb);
}

void RawClient::Restore(const std::string& name, const Callback& cb) {
  client_wrapper_->Restore(name, cb);
}

struct ModelServerContext {
  std::mutex mu;
  ModelServerSplitter splitter;
  int count_down;
  RawClient::Callback cb;
  Status st;
  std::vector<std::unique_ptr<Tensor>> results;
  Tensor* rst;
};

void RawClient::ModelServerForward(int type, const Tensor& ids, Tensor* rst, const Callback& cb) {
  if (type < 1 || type >= client_wrapper_->ServerTypeSize()) {
    cb(Status::ArgumentError("ModelServerForward server type error"));
    return;
  }
  int size = client_wrapper_->ServerSize(type);
  std::unique_ptr<ModelServerContext> ctx(new ModelServerContext);
  ctx->rst = rst;
  ctx->results.resize(size);
  ctx->count_down = size;
  ctx->cb = cb;
  CHECK_ASYNC(ctx->splitter.Init(size, ids));
  std::vector<Tensor> split_id;
  CHECK_ASYNC(ctx->splitter.Split(ids, &split_id));
  ModelServerContext* ctx_ptr = ctx.release();
  for (int i = 0; i < size; i++) {
    client_wrapper_->ModelServerForward(type, i, split_id[i], &ctx_ptr->results[i], [i, ctx_ptr](Status st){
        std::unique_lock<std::mutex> lock(ctx_ptr->mu);
        if (ctx_ptr->st.IsOk() && st.IsOk()) {
          st = ctx_ptr->splitter.Combine(i, *ctx_ptr->results[i], ctx_ptr->rst);
        }
        if (ctx_ptr->st.IsOk() && !st.IsOk()) {
          ctx_ptr->st = st;
          ctx_ptr->cb(st);
        }
        if (--ctx_ptr->count_down == 0) {
          if (ctx_ptr->st.IsOk()) {
            ctx_ptr->cb(Status::Ok());
          }
          lock.unlock();
          delete ctx_ptr;
        }
    });
  }
}

void RawClient::ModelServerBackward(int type, const Tensor& ids, const Tensor& grads, const Callback& cb) {
  if (type < 1 || type >= client_wrapper_->ServerTypeSize()) {
    cb(Status::ArgumentError("ModelServerBackward server type error"));
    return;
  }
  int size = client_wrapper_->ServerSize(type);
  std::unique_ptr<ModelServerContext> ctx(new ModelServerContext);
  ctx->count_down = size;
  ctx->cb = cb;
  CHECK_ASYNC(ctx->splitter.Init(size, ids));
  std::vector<Tensor> split_id;
  std::vector<Tensor> split_grad;
  CHECK_ASYNC(ctx->splitter.Split(ids, &split_id));
  CHECK_ASYNC(ctx->splitter.Split(grads, &split_grad));
  ModelServerContext* ctx_ptr = ctx.release();
  for (int i = 0; i < size; i++) {
    client_wrapper_->ModelServerBackward(type, i, split_id[i], split_grad[i], [ctx_ptr](Status st){
        std::unique_lock<std::mutex> lock(ctx_ptr->mu);
        if (ctx_ptr->st.IsOk() && !st.IsOk()) {
          ctx_ptr->st = st;
          ctx_ptr->cb(st);
        }
        if (--ctx_ptr->count_down == 0) {
          if (ctx_ptr->st.IsOk()) {
            ctx_ptr->cb(Status::Ok());
          }
          lock.unlock();
          delete ctx_ptr;
        }
    });
  }
}

void RawClient::TriggerStreamingModelDense(const std::string& stream_ver, const Callback& cb) {
  client_wrapper_->TriggerStreamingModelDense(stream_ver, cb);
}

void RawClient::TriggerStreamingModelSparse(const std::string& stream_ver, const Callback& cb) {
  client_wrapper_->TriggerStreamingModelSparse(stream_ver, cb);
}

void RawClient::TriggerStreamingModelHash(const std::string& stream_ver, const Callback& cb) {
  client_wrapper_->TriggerStreamingModelHash(stream_ver, cb);
}

void RawClient::AsynchronizeEnter(int id, int staleness, int worker_count, const Callback& cb) {
  client_wrapper_->AsynchronizeEnter(id, staleness, worker_count, cb);
}

void RawClient::SynchronizeEnter(int id, int worker_count, int64_t* token, const Callback& cb) {
  client_wrapper_->SynchronizeEnter(id, worker_count, token, cb);
}

void RawClient::SynchronizeLeave(int id, int64_t token, const Callback& cb) {
  client_wrapper_->SynchronizeLeave(id, token, cb);
}

void RawClient::WorkerReportFinish(int id, const Callback& cb) {
  client_wrapper_->WorkerReportFinish(id, cb);
}

void RawClient::GetWorkerFinishCount(int64_t* count, const Callback& cb) {
  client_wrapper_->GetWorkerFinishCount(count, cb);
}

void RawClient::WorkerBarrier(int id, int worker_count, const Callback& cb) {
  client_wrapper_->WorkerBarrier(id, worker_count, cb);
}


void RawClient::WorkerBarrierV2(int barrier_id, int task_id, int task_num, int token, const Callback& cb) {
  client_wrapper_->WorkerBarrierV2(barrier_id, task_id, task_num, token, cb);    
}

void RawClient::Process(const std::string& var_name, size_t server_id, const UdfChain& udf, const std::vector<Data*>& input, std::vector<Data*>* output, const Callback& cb) {
  client_wrapper_->Process(var_name, server_id, udf.hash(), input, output, [var_name, server_id, udf, input, output, cb, this](Status st) {
    if (st.Code() == Status::kUdfNotRegistered) {
      client_wrapper_->RegisterUdf(server_id, udf, [var_name, server_id, udf, input, output, cb, this](Status st) {
        if (st.IsOk()) {
          client_wrapper_->Process(var_name, server_id, udf.hash(), input, output, cb);
        } else {
          cb(st);
        }
      });
    } else {
      cb(st);
    }
  });
}

Status RawClient::RegisterVariable(const std::string& name, const VariableInfo& info) {
  std::lock_guard<std::mutex> lock(variable_info_mutex_);
  auto iter = args_.variable_info.find(name);
  if (iter != args_.variable_info.end()) {
    return Status::Ok();
  }
  args_.variable_info[name] = info;
  args_.variable_info[name].visit_time = 0;
  args_.variable_info[name].dense_visit_ids = 0;
  args_.variable_info[name].sparse_visit_ids = 0;
  init_variable_info_ = false;
  return Status::Ok();
}

Status RawClient::UpdateVariableVisitInfo(const std::string& name, int64_t id_num) {
  std::lock_guard<std::mutex> lock(variable_info_mutex_);
  std::string realname = name;
  if (!realname.empty() && realname[0] == '^') {
    realname = realname.substr(1);
  }
  std::promise<Status> st_promise;
  client_wrapper_->UpdateVariableVisitInfo(realname, id_num, [&st_promise](Status st){
    st_promise.set_value(st);
  });
  Status st = st_promise.get_future().get();
  return st;
}

Status RawClient::GetVariableInfo(const std::string& name, VariableInfo* info) {
  std::lock_guard<std::mutex> lock(variable_info_mutex_);
  if (!init_variable_info_) {
    std::vector<VariableInfo> inputs;
    std::vector<VariableInfo> outputs;
    for (const auto& item : args_.variable_info) {
      inputs.push_back(item.second);
    }
    std::promise<Status> st_promise;
    client_wrapper_->UpdateVariableInfo(inputs, &outputs, [&st_promise](Status st){
      st_promise.set_value(st);
    });
    Status st = st_promise.get_future().get();
    if (!st.IsOk()) {
      return st;
    }

    variable_infos_.clear();
    for (const auto& item : outputs) {
      variable_infos_[item.name] = item;
    }
    init_variable_info_ = true;

    std::string logger = "";
    for (const auto& item : outputs) {
      logger += item.name + "<shape[";
      for (auto dim : item.shape) {
          logger += std::to_string(dim) + ",";
      }
      logger += "]parts[";
      for (auto part : item.parts) {
          logger += std::to_string(part.server) + ":" + std::to_string(part.size) + ",";
      }
      logger += "]>";
    }
    LOG(INFO) << "Variable Info: " << logger;
  }

  std::string realname = name;
  if (!realname.empty() && realname[0] == '^') {
    realname = realname.substr(1);
  }
  auto iter = variable_infos_.find(realname);
  if (iter == variable_infos_.end()) {
    return Status::ArgumentError("Variable " + realname + " Not Found In Variable Info");
  }
  *info = iter->second;
  return Status::Ok();
}

} //namespace client
} //namespace ps


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

#include "scheduler_impl.h"

#include <atomic>
#include <chrono>
#include <future>
#include <sstream>
#include <cstdlib>

#include "ps-plus/common/logging.h"
#include "ps-plus/common/string_utils.h"
#include "ps-plus/common/file_system.h"
#include "ps-plus/common/serializer.h"
#include "ps-plus/common/initializer/none_initializer.h"

#define CHECK_COND(cond, str) do {if (!(cond)) {return Status::ArgumentError(str);}} while(0);

using namespace ps;
using namespace ps::scheduler;
using namespace std;
using namespace std::chrono;

SchedulerImpl::SchedulerImpl(
    const string& server_count,
    const string& scheduler_addr,
    const string& checkpoint_path,
    const Placementer::Arg& placement_arg,
    const string& streaming_dense_model_addr,
    const string& streaming_sparse_model_addr,
    const string& streaming_hash_model_addr,
    bool bind_cores)
    : main_thread_(nullptr), meta_thread_(nullptr), stopped_(false), ready_(false),
      version_(kUnusedVersion),
      server_count_(server_count),
      checkpoint_path_(checkpoint_path),
      placement_arg_(placement_arg),
      streaming_dense_model_addr_(streaming_dense_model_addr),
      streaming_sparse_model_addr_(streaming_sparse_model_addr),
      streaming_hash_model_addr_(streaming_hash_model_addr) {
  service_.reset(new SchedulerService(this, server_count, scheduler_addr, bind_cores));
  char* vp_var = std::getenv("vp_method");
  char* meta_var = std::getenv("meta_dir");
  if (vp_var != NULL ) { vp_string_ = vp_var; }
  if (meta_var != NULL) { meta_string_ = meta_var; }
  else { meta_string_ = ""; }
  if (vp_string_ == "balance") { placementer_ = GetPlugin<Placementer>("Balance"); }
  else { placementer_ = GetPlugin<Placementer>("BalanceV2"); }
  lazy_queue_.reset(new ThreadPool(1));
  synchronizer_queue_.reset(new ThreadPool(1));
}

SchedulerImpl::~SchedulerImpl() {
  if (main_thread_) {
    stopped_ = true;
    main_thread_->join();
    if (vp_string_ == "anneal" && !meta_string_.empty()) { meta_thread_->join(); }
  }
}

Status SchedulerImpl::Start() {
  if (!streaming_dense_model_addr_.empty()) {
    PS_CHECK_STATUS(StreamingModelManager::OpenWriterAny(streaming_dense_model_addr_, &streaming_dense_model_writer_));
  }
  main_thread_.reset(new thread(&SchedulerImpl::Main, this));
  if (vp_string_ == "anneal" && !meta_string_.empty()) { 
    meta_thread_.reset(new thread(&SchedulerImpl::WriteMetaInfo, this)); 
  }
  service_->Start();
  LOG(INFO) << "Started scheduler main thread";
  return Status::Ok();
}

Version SchedulerImpl::GetVersion() {
  unique_lock<mutex> lock(m_);
  return ready_ ? version_ : kUnusedVersion;
}

Status SchedulerImpl::RegisterServer(const ServerInfo& server) {
  unique_lock<mutex> lock(m_);
  if (disconnected_server_.find(server) != disconnected_server_.end()) {
    LOG(INFO) << "Skip Disconnected Server " << server.ToString();
    return Status::ArgumentError("you are died.");
  }
  const std::pair<ServerType, ServerId> id(server.GetServerType(), server.GetId());
  const auto& it = servers_.find(id);
  if (it == servers_.end()) {
    servers_[id] = server;
    service_->SetServer(id.first, id.second, server.Address());
    LOG(INFO) << "Added new server " << server.ToString();
  } else {
    ServerInfo old_server = it->second;
    if (old_server != server) {
      ready_ = false;
      r_ready_ = false;
      servers_[id] = server;
      service_->SetServer(id.first, id.second, server.Address());
      LOG(INFO) << "Server" << old_server.ToString() << " failed And Restore at " << server.ToString();
      disconnected_server_.insert(old_server);
      op_cv_.notify_all();
    }
  }
  return Status::Ok();
}

Status SchedulerImpl::GetClusterInfo(const Version version, ClusterInfo* result) {
  unique_lock<mutex> lock(m_);
  if (!ready_) { return Status::NotReady("Cluster is not ready"); }
  if (version != version_) { return VersionMismatch(version_, version); }
  for (const auto& it: servers_) { result->AddServer(it.second); }
  for (int i = 0; i < service_->GetServerTypeSize(); i++) {
    result->server_size_.push_back(service_->GetServerSize(i));
  }
  return Status::Ok();
}

void SchedulerImpl::Save(Version version, const string& checkpoint,
                         OpCallback cb) {
  AssignOp(kSave, version, checkpoint, cb);
}

void SchedulerImpl::Restore(Version version, const string& checkpoint,
                            OpCallback cb) {
  AssignOp(kRestore, version, checkpoint, cb);
}

void SchedulerImpl::TriggerStreamingDense(Version version, const std::string& stream_version, OpCallback cb) {
  lazy_queue_->Schedule([=](){cb(InternalTriggerStreamingDense(version, stream_version));});
}

Status SchedulerImpl::InitGlobalQueue(
    Version version, 
    const std::string& name,
    const std::vector<std::string>& paths,
    size_t epochs,
    bool epoch_isolate) {
  std::unique_lock<std::mutex> lock(mu_);
  auto it = global_file_queues_.find(name);
  if (it == global_file_queues_.end()) {
    global_file_queues_[name].reset(new GlobalFileQueue());
  }

  return global_file_queues_[name]->Init(
      paths, epochs, epoch_isolate);
}

Status SchedulerImpl::GetNextFile(
    Version version, 
    const std::string& name,
    size_t worker_id, 
    WorkerState* ws) {
  std::unique_lock<std::mutex> lock(mu_);
  return global_file_queues_[name]->GetNextFile(
      worker_id, ws);
}

Status SchedulerImpl::ReportWorkerState(
    Version version, 
    const std::string& name,
    size_t worker_id,
    const std::vector<ps::WorkerState>& worker_states) {
  std::unique_lock<std::mutex> lock(mu_);
  return global_file_queues_[name]->ReportWorkerState(
      worker_id, worker_states);
}

Status SchedulerImpl::RestoreWorkerState(
    Version version, 
    const std::string& name,
    size_t worker_id) {
  std::unique_lock<std::mutex> lock(mu_);
  return global_file_queues_[name]->RestoreWorkerState(
      worker_id);
}

void SchedulerImpl::TriggerStreamingSparse(Version version, const std::string& stream_version, OpCallback cb) {
  lazy_queue_->Schedule([=](){cb(InternalTriggerStreamingSparse(version, stream_version));});
}

void SchedulerImpl::TriggerStreamingHash(Version version, const std::string& stream_version, OpCallback cb) {
  lazy_queue_->Schedule([=](){cb(InternalTriggerStreamingHash(version, stream_version));});
}

void SchedulerImpl::AsynchronizeEnter(Version version, int id, int staleness, int worker_count,
                                     function<void (const Status&)> cb) {
  synchronizer_queue_->Schedule([=] {
    InternalAsynchronizeEnter(version, id, staleness, worker_count, cb);
  });
}

void SchedulerImpl::SynchronizeEnter(Version version, int id, int worker_count, function<void (int64_t, const Status&)> cb) {
  synchronizer_queue_->Schedule([=] {
    InternalSynchronizeEnter(version, id, worker_count, cb);
  });
}

void SchedulerImpl::SynchronizeLeave(Version version, int id, int64_t token, function<void (const Status&)> cb) {
  synchronizer_queue_->Schedule([=] {
    InternalSynchronizeLeave(version, id, token, cb);
  });
}

void SchedulerImpl::WorkerReportFinish(Version version, int id, function<void (const Status&)> cb) {
  synchronizer_queue_->Schedule([=] {
    InternalWorkerReportFinish(version, id, cb);
  });
}

void SchedulerImpl::WorkerBarrier(Version version, int id, int worker_count, function<void (const Status&)> cb) {
  synchronizer_queue_->Schedule([=] {
    InternalWorkerBarrier(version, id, worker_count, cb);
  });
}

void SchedulerImpl::WorkerBarrierV2(
    Version version, 
    int barrier_id, 
    int task_id, 
    int task_num,
    int token,
    function<void (const Status&)> cb) {
  synchronizer_queue_->Schedule([=] {
    InternalWorkerBarrierV2(version, barrier_id, task_id, task_num, token, cb);
  });
}

void SchedulerImpl::GetWorkerFinishCount(Version version, function<void (int64_t, const Status&)> cb) {
  synchronizer_queue_->Schedule([=] {
    InternalGetWorkerFinishCount(version, cb);
  });
}

Status SchedulerImpl::UpdateVariableInfo(Version version,
                                         const vector<VariableInfo>& info,
                                         vector<VariableInfo>* result) {
  unique_lock<mutex> lock(m_);
  if (!ready_) { return Status::NotReady("Cluster is not ready"); }
  if (version != version_) { return VersionMismatch(version_, version); }
  return InternalUpdateVariableInfo(info, result);
}

Status SchedulerImpl::InternalUpdateVariableInfo(const vector<VariableInfo>& info, vector<VariableInfo>* result) {
  map<string, VariableInfo> m;
  for (const auto& i: info) { m[i.name] = i; }
  for (auto& i: variable_info_) {
    if (m[i.name].type == VariableInfo::Type::kHash128 || m[i.name].type == VariableInfo::Type::kHash64) {
      i.shape[0] = m[i.name].shape[0];
    }
  }
  for (const auto& i: variable_info_) { m[i.name] = i; }

  // fill in new args
  std::vector<std::string> arg_names{"save_ratio", "batch_read", "mem_ratio", "no_split", "save", VariableInfo::ORIGIN_NAME, VariableInfo::ORIGIN_FILE_PATH};
  for (const auto& i: info) {
    for (auto& arg_name: arg_names) {
      auto it = i.args.find(arg_name);
      if (it != i.args.end()) m[i.name].args[arg_name] = it->second;
    }
  }

  vector<VariableInfo> v;
  for (const auto& it: m) { v.push_back(it.second); }
  const auto& st = placementer_->Placement(v, result, placement_arg_, service_->GetServerSize(0));
  if (!st.IsOk()) { return st; }
  variable_info_.clear();
  for (const auto& i: *result) { variable_info_.push_back(i); }
  return Status::Ok();    
}

Status SchedulerImpl::UpdateVariableVisitInfo(Version version, const string& var_name, int64_t ids) {
  unique_lock<mutex> lock(m_);
  if (!ready_) { return Status::NotReady("Cluster is not ready"); }
  if (version != version_) { return VersionMismatch(version_, version); }
  for (auto& i : variable_info_) {
    if (i.name == var_name) { 
      i.visit_time++;
      if (ids < 0) {
		if (i.shape.empty()) { i.dense_visit_ids += 1; }
		else { i.dense_visit_ids += i.shape[0]; }
      } else {
        i.sparse_visit_ids += ids;
      }
      break;
    }
  }
  return Status::Ok();
}

void SchedulerImpl::Main() {
  while (!stopped_) {
    WaitForServers();
    LOG(INFO) << "All servers are ready";
    {
      unique_lock<mutex> lock(m_);
      op_code_ = kRestore;
      op_checkpoint_ = "";
      op_cb_ = [](const ps::Status&){};
      LOG(INFO) << "Try Restoring to latest checkpoint due to error";
    }
    MainLoop();
  }
}

Status SchedulerImpl::WriteMetaInfo() {
  size_t max_visit = 0;
  while (true) {
    for (auto& info : variable_info_) {
      if (info.visit_time > max_visit) { max_visit = info.visit_time; }
    }
    if (max_visit > 2000) { break; }
    this_thread::sleep_for(seconds(1));
  }
  
  {
    unique_lock<mutex> lock(m_);
    stopped_ = true;
    op_cv_.notify_all();
    LOG(INFO) << "Write Placement Meta Info To: " << meta_string_;
    std::unique_ptr<FileSystem::WriteStream> s;
    PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(meta_string_, &s));
    PS_CHECK_STATUS(s->WriteRaw(variable_info_.size()));
    for (auto& info : variable_info_) {
	  string data;
	  if (info.shape.empty()) {
	    data = info.name + " " + std::to_string(info.visit_time) + " "
             + std::to_string(info.dense_visit_ids) + " " + std::to_string(info.sparse_visit_ids) + " "
             + std::to_string(1) + "\n";	
	  } else {
	    data = info.name + " " + std::to_string(info.visit_time) + " "
             + std::to_string(info.dense_visit_ids) + " " + std::to_string(info.sparse_visit_ids) + " "
             + std::to_string(info.shape[0]) + "\n";
	  }
      PS_CHECK_STATUS(s->WriteStr(data));
    }
  }
  return Status::Ok();
}

void SchedulerImpl::MainLoop() {
  while (!stopped_) {
    // First Run: Must Have a restore op
    switch (op_code_) {
    case kSave: {
      LOG(INFO) << "Saving checkpoint " << op_checkpoint_;
      Status st = InternalSave(op_checkpoint_);
      LOG(INFO) << "Saving checkpoint " << op_checkpoint_ << ", Get Status " << st.ToString();
      op_cb_(st);
      break;
    }
    case kRestore: {
      LOG(INFO) << "Restore checkpoint " << op_checkpoint_;
      Status st = InternalRestore(op_checkpoint_);
      LOG(INFO) << "Restore checkpoint " << op_checkpoint_ << ", Get Status " << st.ToString();
      op_cb_(st);
      if (sync_) { sync_->Reset(); }
      break;
    }
    default: {
      LOG(FATAL) << "Invalid op code " << op_code_;
      abort();
    }
    }
    {
      unique_lock<mutex> lock(m_);
      op_code_ = kNone;
      op_checkpoint_ = "";
      op_cb_ = [](const Status&){
        LOG(FATAL) << "Invalid Op Call for NoneOp";
        abort();
      };
    }

    WaitForOp();
    if (stopped_ || !ready_) {
      return;
    }
  }
}

Status SchedulerImpl::VersionMismatch(Version exp, Version act) {
  ostringstream oss;
  oss << "Version mismatch: expected=" << exp << ", actual=" << act;
  return Status::VersionMismatch(oss.str());
}

string SchedulerImpl::OpName(OpCode code) {
  switch (code) {
  case kSave:    return "save";
  case kRestore: return "restore";
  default: {
    LOG(FATAL) << "Invalid op code " << code;
    abort();
  }
  }
}

void SchedulerImpl::WaitForOp() {
  unique_lock<mutex> lock(m_);
  if (op_code_ != kNone || !ready_ || stopped_) {
    return;
  }
  op_cv_.wait(lock, [&] { return op_code_ != kNone || !ready_ || stopped_; });
}

void SchedulerImpl::AssignOp(OpCode code, Version version,
                             const string& checkpoint, OpCallback cb) {
  unique_lock<mutex> lock(m_);
  if (!ready_) {
    lock.unlock();
    cb(Status::NotReady("Cluster is not ready"));
    return;
  }
  if (version != version_) {
    lock.unlock();
    cb(VersionMismatch(version_, version));
    return;
  }
  if (op_code_ != kNone) {
    ostringstream oss;
    oss << "The current op is to " << OpName(op_code_) << " checkpoint `"
        << op_checkpoint_ << "`, not assigning new op to "
        << OpName(code) << " checkpoint `" << checkpoint << "`";
    LOG(INFO) << oss.str();
    cb(Status::ConcurrentExecution(oss.str()));
  } else {
    op_code_ = code;
    op_checkpoint_ = checkpoint;
    op_cb_ = cb;
    op_cv_.notify_all();
    LOG(INFO) << "Schedule a new op to " << OpName(code) << " checkpoint " << checkpoint;
  }
}

void SchedulerImpl::WaitForServers() {
  while (true) {
    {
      unique_lock<mutex> lock(m_);
      const size_t n = service_->GetServerTotalSize() - servers_.size();
      if (n == 0) { return; }
      LOG(INFO) << "Waiting for " << n << " more server";
    }
    this_thread::sleep_for(seconds(1));
  }
}

Status SchedulerImpl::InternalRestore(const string& checkpoint) {
  std::promise<Status> result;
  std::mutex mu;
  Status collect;
  size_t count_down;
  {
    unique_lock<mutex> lock(m_);

    // disable
    ready_ = false;
    r_ready_ = true;
    version_ = NewRandomVersion();

    vector<string> checkpoints;
    PS_CHECK_STATUS(ReadCheckpoints(checkpoint_path_, true, &checkpoints));
    string real_checkpoint = "";
    if (checkpoint == "") {
      real_checkpoint = checkpoints.size() == 0 ? "" : checkpoints.back();
    } else {
      for (auto&& ck : checkpoints) {
        if (checkpoint == ck) {
          real_checkpoint = checkpoint;
          break;
        }
      }
      if (real_checkpoint == "") {
        return Status::NotFound("Checkpoint Not Found : " + checkpoint);
      }
    }
    LOG(INFO) << "Real checkpoint is " << real_checkpoint;
    
    vector<VariableInfo> source_infos;
    PS_CHECK_STATUS(GenerateVariableInfo(real_checkpoint, &source_infos));
    PS_CHECK_STATUS(RestoreGlobalQueue(checkpoint_path_ + "/" + real_checkpoint));

    count_down = service_->GetServerTotalSize();
    LOG(INFO) << "VariableInfos " << PrintVariableInfo(variable_info_);
    for (const auto& it: servers_) {
      const ServerInfo& server = it.second;
      const ServerId id = server.GetId();
      ServerType server_type = server.GetServerType();
      if (server_type == 0) {
        service_->ServerRestore(
            server_type, id, version_,
            source_infos, variable_info_, [id, &result, &mu, &collect, &count_down](Status st) {
              std::unique_lock<std::mutex> lock(mu);
              if (!st.IsOk() && collect.IsOk()) {
                collect = st;
              }
              if (--count_down == 0) {
                lock.unlock();
                result.set_value(collect);
              }
              LOG(INFO) << "server " << id << " finish restore, status " << (st.IsOk() ? "OK" : st.Msg()) << ", waiting " << count_down << " more";
            });
      } else {
        service_->ModelServerFlush(
            server_type, id, version_,
            [&result, &mu, &collect, &count_down](Status st) {
              std::unique_lock<std::mutex> lock(mu);
              if (!st.IsOk() && collect.IsOk()) {
                collect = st;
              }
              if (--count_down == 0) {
                lock.unlock();
                result.set_value(collect);
              }
            });
      }
    }
  }
  result.get_future().wait();
  if (collect.IsOk()) {
    // enable
    unique_lock<mutex> lock(m_);
    ready_ = r_ready_;
  }
  return collect;
}

Status SchedulerImpl::GenerateVariableInfo(string real_checkpoint, vector<VariableInfo>* source) {
  std::string checkpoint = "";
  if (real_checkpoint != "") {
    checkpoint += "*@" + checkpoint_path_ + "/" + real_checkpoint;
  }
  if (checkpoint != "") {
    vector<VariableInfo> result;
    PS_CHECK_STATUS(ParseVariables(checkpoint, source));
    vector<VariableInfo> input_infos = *source;
    for (auto& i: input_infos) {
      i.parts.clear();
    }
    PS_CHECK_STATUS(InternalUpdateVariableInfo(input_infos, &result));
    variable_info_ = result;
  } else {
    LOG(INFO) << "Empty checkpoint, not load";
  }
  return Status::Ok();
}

Status SchedulerImpl::ParseVariables(const std::string& checkpoint, vector<VariableInfo>* result) {
  map<string, VariableInfo> var_map;
  vector<string> v = StringUtils::split(checkpoint, "@");
  string source_path = v[1];
  if (source_path.back() == '/') {
    source_path = source_path.substr(0, source_path.size()-1);
  }
  vector<string> checkpoints;
  PS_CHECK_STATUS(ReadCheckpoints(source_path, true, &checkpoints));
  if (checkpoints.size() != 0) {
    source_path += "/" + checkpoints.back();
  }
  size_t server_num;
  std::vector<VariableInfo> infos;
  PS_CHECK_STATUS(ReadVariableInfoMeta(source_path, &server_num, &infos));
  for (VariableInfo& info : infos) {
    info.args[VariableInfo::ORIGIN_FILE_PATH] = source_path;
    info.args[VariableInfo::ORIGIN_NAME] = info.name;
    var_map[info.name] = info;
  }
  for (auto& iter : var_map) {
    result->push_back(iter.second);
  }
  return Status::Ok();
}

string SchedulerImpl::PrintVariableInfo(const vector<VariableInfo>& infos) {
  string logger = "";
  for (const auto& item : infos) {
    logger += item.name + "<shape[";
    for (auto dim : item.shape) {
      logger += std::to_string(dim) + ",";
    }
    logger += "]parts[";
    for (auto part : item.parts) {
      logger += std::to_string(part.server) + ":" + std::to_string(part.size) + ",";
    }
    logger += "]>args[";
    for (auto& iter : item.args) {
      logger += iter.first + "=" + iter.second + ",";
    }
    logger += "]";
  }
  return logger;
}    

Status SchedulerImpl::InternalSave(const string& checkpoint) {
  std::promise<Status> result;
  std::mutex mu;
  Status collect;
  size_t count_down;
  {
    unique_lock<mutex> lock(m_);
    if (!ready_) {
      Status ret = Status::NotReady("Error saving to " + checkpoint + "` : cluster not ready");
      LOG(ERROR) << ret.ToString();
      return ret;
    }

    {
      std::unique_ptr<FileSystem::WriteStream> s;
      PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(checkpoint_path_ + "/" + checkpoint + "/__meta__", &s));

      {
        string infos_buf;
        size_t infos_type;
        std::unique_ptr<WrapperData<VariableInfoCollection>> info_wrapper(new WrapperData<VariableInfoCollection>);
        for (const auto& info: variable_info_) {
          auto iter = info.args.find("save");
          if (iter != info.args.end() && iter->second == "false") {
            continue;
          }
          info_wrapper->Internal().infos.push_back(info);
        }
        vector<serializer::Fragment> frags;
        serializer::MemGuard mem;
        PS_CHECK_STATUS(serializer::SerializeAny<Data>(info_wrapper.get(), &infos_type, &frags, mem));
        for (auto frag : frags) {
          infos_buf.append(frag.base, frag.size);
        }
        PS_CHECK_STATUS(s->WriteRaw((size_t)service_->GetServerSize(0)));
        PS_CHECK_STATUS(s->WriteRaw(infos_type));
        PS_CHECK_STATUS(s->WriteStr(infos_buf));
      }
    }

    count_down = service_->GetServerSize(0);

    for (auto& it: servers_) {
      ServerInfo& server = it.second;
      ServerId id = server.GetId();
      ServerType server_type = server.GetServerType();
      if (server_type == 0) {
        service_->ServerSave(server_type, id, version_, checkpoint_path_ + "/" + checkpoint, variable_info_, [id, &result, &mu, &collect, &count_down](Status st) {
              std::unique_lock<std::mutex> lock(mu);
              if (!st.IsOk() && collect.IsOk()) {
                collect = st;
              }
              if (--count_down == 0) {
                lock.unlock();
                result.set_value(collect);
              }
              LOG(INFO) << "server " << id << " finish save, status " << (st.IsOk() ? "OK" : st.Msg()) << ", waiting " << count_down << " more.";
            });
      }
    }
  }
  result.get_future().wait();
  PS_CHECK_STATUS(collect);

  vector<string> checkpoints;
  {
    std::unique_ptr<FileSystem::ReadStream> s;
    Status st = FileSystem::OpenReadStreamAny(checkpoint_path_ + "/checkpoints", &s);
    // Ignore st fail when we use a fresh checkpoint dir
    if (st.IsOk()) {
      size_t size;
      PS_CHECK_STATUS(s->ReadRaw(&size));
      checkpoints.resize(size);
      for (size_t i = 0; i < size; i++) {
        PS_CHECK_STATUS(s->ReadStr(&checkpoints[i]));
      }
    }
  }
  checkpoints.push_back(checkpoint);
  {
    std::unique_ptr<FileSystem::WriteStream> s;
    PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(checkpoint_path_ + "/checkpoints.tmp", &s));
    size_t size = checkpoints.size();
    PS_CHECK_STATUS(s->WriteRaw(size));
    for (size_t i = 0; i < size; i++) {
      PS_CHECK_STATUS(s->WriteStr(checkpoints[i]));
    }
  }
  FileSystem::RemoveAny(checkpoint_path_ + "/checkpoints");
  PS_CHECK_STATUS(FileSystem::RenameAny(checkpoint_path_ + "/checkpoints.tmp", checkpoint_path_ + "/checkpoints"));

  {
    std::unique_ptr<FileSystem::WriteStream> s;
    PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(checkpoint_path_ + "/" + checkpoint + "/global_queue_meta", &s));
    std::string queue_buf;
    PS_CHECK_STATUS(SerializeGlobalQueue(&queue_buf));
    PS_CHECK_STATUS(s->WriteStr(queue_buf));
  }

  return Status::Ok();;
}

Status SchedulerImpl::InternalTriggerStreamingDense(Version version, const std::string& stream_version) {
  vector<ps::VariableInfo> variable_info;
  {
    unique_lock<mutex> lock(m_);
    if (!ready_) {
      return Status::NotReady("Cluster is not ready");
    }
    if (version != version_) {
      return VersionMismatch(version_, version);
    }
    variable_info = variable_info_;
  }
  if (streaming_dense_model_addr_.empty()) {
    return Status::Unknown("Dense Model Addr is Empty, Dense Model is disable");
  }
  if (streaming_dense_model_writer_ == nullptr) {
    return Status::Unknown("Dense Model Writer Open Error");
  }
  std::unordered_set<string> dense_vars;
  for (auto& info : variable_info) {
    if (info.args["streaming_dense_output"] == "true") {
      dense_vars.insert(info.name);
    }
  }
  {
    std::promise<Status> result;
    std::mutex mu;
    Status collect;
    std::atomic<size_t> count_down(service_->GetServerSize(0));
    {
      unique_lock<mutex> lock(m_);
      for (auto& it: servers_) {
        ServerInfo& server = it.second;
        ServerId id = server.GetId();
        ServerType server_type = server.GetServerType();
        if (server_type == 0) {
          service_->ServerStreamingDenseVarName(
              server_type, id, version,
              [&, id](Status st, const DenseVarNames& vars) {
                std::unique_lock<std::mutex> lock(mu);
                if (!st.IsOk() && collect.IsOk()) {
                  collect = st;
                } else {
                  for (auto& item : vars.names) {
                    dense_vars.insert(item);
                  }
                }
                if (--count_down == 0) {
                  lock.unlock();
                  result.set_value(collect);
                }
              });
        }
      }
    }
    result.get_future().wait();
    PS_CHECK_STATUS(collect);
  }
  DenseVarNames vars;
  for (auto& item : dense_vars) {
    vars.names.push_back(item);
  }
  std::unordered_map<string, vector<DenseVarValues::DenseVarValue>> values;
  { 
    std::promise<Status> result;
    std::mutex mu;
    Status collect;
    std::atomic<size_t> count_down(service_->GetServerSize(0));
    {
      unique_lock<mutex> lock(m_);
      for (auto& it: servers_) {
        ServerInfo& server = it.second;
        ServerId id = server.GetId();
        ServerType server_type = server.GetServerType();
        if (server_type == 0) {
          service_->ServerGatherStreamingDenseVar(
              server_type, id, version, vars,
              [&count_down, &mu, &result, &collect, &values, id](Status st, const DenseVarValues& val) {
                std::unique_lock<std::mutex> lock(mu);
                if (!st.IsOk() && collect.IsOk()) {
                  collect = st;
                } else {
                  for (auto& item : val.values) {
                    Tensor copy_tensor(item.data.Type(), item.data.Shape(), new initializer::NoneInitializer);
                    memcpy(copy_tensor.Raw<char>(), item.data.Raw<char>(), SizeOfType(copy_tensor.Type()) * copy_tensor.Shape().NumElements());
                    values[item.name].push_back(DenseVarValues::DenseVarValue{
                          .name = item.name,
                          .offset = item.offset,
                          .data = copy_tensor
                            });
                  }
                }
                if (--count_down == 0) {
                  lock.unlock();
                  result.set_value(collect);
                }
              });
        }
      }
    }
    result.get_future().wait();
    PS_CHECK_STATUS(collect);
  }
  std::unordered_map<string, Tensor> datas;
  for (auto& item : values) {
    string name = item.first;
    vector<DenseVarValues::DenseVarValue> val = item.second;
    VariableInfo info;
    bool found = false;
    for (auto& xinfo : variable_info) {
      if (xinfo.name == name) {
        found = true;
        info = xinfo;
        break;
      }
    }
    if (!found) {
      return Status::NotFound("Not Found variable info: " + name);
    }
    Tensor result(info.datatype, TensorShape(vector<size_t>(info.shape.begin(), info.shape.end())), new initializer::NoneInitializer);
    size_t slice_size = SizeOfType(info.datatype);
    for (size_t i = 1; i < info.shape.size(); i++) {
      slice_size *= info.shape[i];
    }
    std::sort(val.begin(), val.end(),
        [](const DenseVarValues::DenseVarValue& lhs, const DenseVarValues::DenseVarValue& rhs) {
      return lhs.offset < rhs.offset;
    });
    size_t offset = 0;
    for (auto& item : val) {
      if (item.data.Shape().Size() != info.shape.size()) {
        return Status::Unknown("Error Shape on cobime variable: " + name);
      }
      for (size_t i = 1; i < info.shape.size(); i++) {
        if ((size_t)info.shape[i] != item.data.Shape()[i]) {
          return Status::Unknown("Error Shape on cobime variable: " + name);
        }
      }
      if (item.data.Type() != info.datatype) {
        return Status::Unknown("Error Type on cobime variable: " + name);
      }
      if (offset != item.offset) {
        return Status::Unknown("Error Offset on cobime variable: " + name);
      }
      memcpy(result.Raw<char>() + offset * slice_size, item.data.Raw<char>(), item.data.Shape().NumElements() * SizeOfType(item.data.Type()));
      offset += item.data.Shape().Size() > 0 ? item.data.Shape()[0] : 0;
    }
    if (info.shape.size() > 0 && (size_t)info.shape[0] != offset) {
      return Status::Unknown("Error Total Offset on cobime variable: " + name);
    }
    datas[name] = result;
  }
  vector<StreamingModelWriter::DenseModel> result;
  for (auto& item : datas) {
    result.push_back(StreamingModelWriter::DenseModel{.name = item.first, .data = item.second});
  }
  Status rst = streaming_dense_model_writer_->WriteDenseModel(result, stream_version);
  return rst;
}

Status SchedulerImpl::InternalTriggerStreamingSparse(Version version, const std::string& stream_version) {
  {
    unique_lock<mutex> lock(m_);
    if (!ready_) {
      return Status::NotReady("Cluster is not ready");
    }
    if (version != version_) {
      return VersionMismatch(version_, version);
    }
  }
  {
    std::promise<Status> result;
    std::mutex mu;
    Status collect;
    std::atomic<size_t> count_down(service_->GetServerSize(0));
    {
      unique_lock<mutex> lock(m_);
      for (auto& it: servers_) {
        ServerInfo& server = it.second;
        ServerType server_type = server.GetServerType();
        ServerId id = server.GetId();
        if (server_type == 0) {
          service_->ServerTriggerStreamingSparse(
              server_type, id, version, stream_version,
              [id, &result, &mu, &collect, &count_down](Status st) {
                std::unique_lock<std::mutex> lock(mu);
                if (!st.IsOk() && collect.IsOk()) {
                  collect = st;
                }
                if (--count_down == 0) {
                  lock.unlock();
                  result.set_value(collect);
                }
              });
        }
      }
    }
    result.get_future().wait();
    PS_CHECK_STATUS(collect);
  }
  return Status::Ok();
}

Status SchedulerImpl::InternalTriggerStreamingHash(Version version, const std::string& stream_version) {
  {
    unique_lock<mutex> lock(m_);
    if (!ready_) {
      return Status::NotReady("Cluster is not ready");
    }
    if (version != version_) {
      return VersionMismatch(version_, version);
    }
  }
  {
    std::promise<Status> result;
    std::mutex mu;
    Status collect;
    std::atomic<size_t> count_down(service_->GetServerSize(0));
    {
      unique_lock<mutex> lock(m_);
      for (auto& it: servers_) {
        ServerInfo& server = it.second;
        ServerId id = server.GetId();
        ServerType server_type = server.GetServerType();
        if (server_type == 0) {
          service_->ServerTriggerStreamingHash(
              server_type, id, version, stream_version,
              [id, &result, &mu, &collect, &count_down](Status st) {
                std::unique_lock<std::mutex> lock(mu);
                if (!st.IsOk() && collect.IsOk()) {
                  collect = st;
                }
                if (--count_down == 0) {
                  lock.unlock();
                  result.set_value(collect);
                }
              });
        }
      }
    }
    result.get_future().wait();
    PS_CHECK_STATUS(collect);
  }
  return Status::Ok();
}

void SchedulerImpl::InternalAsynchronizeEnter(Version version, int id, int staleness, int worker_count, function<void (const Status&)> cb) {
  if (version != version_) {
    cb(VersionMismatch(version_, version));
    return;
  }
  if (!sync_) {
    auto sync = new Asynchronizer(staleness, worker_count);
    sync_.reset(sync);
  }
  auto sync = dynamic_cast<Asynchronizer*>(sync_.get());
  if (sync == nullptr) {
    LOG(ERROR) << "Call Async method in sync mode.";
    cb(Status::ArgumentError("Call Async method in sync mode."));
  }
  sync->Enter(id, cb);
}

void SchedulerImpl::InternalWorkerReportFinish(Version version, int id, function<void (const Status&)> cb) {
  if (version != version_) {
    cb(VersionMismatch(version_, version));
    return;
  }
  if (sync_.get() != nullptr) {
    Status st = sync_->WorkerReportFinish(id);
    if (!st.IsOk()) {
      cb(st);
      return;
    }
  }

  finished_workers_.insert(id);

  {
    auto iter = worker_barriers_.find(id);
    if (iter != worker_barriers_.end()) {
      worker_barriers_.erase(iter);
    }

    if (worker_barriers_.size() == worker_count_ - finished_workers_.size()) {
      for (auto iter : worker_barriers_) {
        (iter.second)(Status::Ok());
      }

      worker_barriers_.clear();
    }
  }

  {
    auto iter = barrier_infos_.find(id);
    if (iter != barrier_infos_.end()) {
      barrier_infos_.erase(iter);
    }

    if (barrier_infos_.size() == worker_count_ - finished_workers_.size()) {
      for (auto iter : barrier_infos_) {
        (iter.second.cb)(Status::Ok());
      }

      barrier_infos_.clear();
    }
  }

  cb(Status::Ok());
}

void SchedulerImpl::InternalWorkerBarrier(Version version, int id, int worker_count, function<void (const Status&)> cb) {
  worker_count_ = worker_count;
  if (version != version_) {
    cb(VersionMismatch(version_, version));
    return;
  }
  worker_barriers_[id] = cb;
  if (worker_barriers_.size() == worker_count - finished_workers_.size()) {
    for (auto iter : worker_barriers_) {
      (iter.second)(Status::Ok());
    }
    worker_barriers_.clear();
  }
}

void SchedulerImpl::InternalWorkerBarrierV2(
    Version version, 
    int barrier_id, 
    int task_id, 
    int task_num,
    int token,
    function<void (const Status&)> cb) {
  worker_count_ = task_num;
  if (version != version_) {
    cb(VersionMismatch(version_, version));
    return;
  }

  auto it = barrier_tokens_.find(task_id);
  if (it == barrier_tokens_.end()) {
    barrier_tokens_[task_id] = token;
  }

  BarrierV2Info bi = BarrierV2Info{
    .token=token, 
    .barrier_id=barrier_id, 
    .cb=cb};

  if (barrier_tokens_[task_id] == token) {
    if (barrier_infos_.empty()) {
      for (auto& it: failover_barrier_infos_) {
        if (it.second.barrier_id == barrier_id) {
          barrier_infos_[it.first] = it.second;
        } else {
          (it.second.cb)(Status::Ok());
        }
      }

      failover_barrier_infos_.clear();
    }

    BarrierAddAndNotifyAll(task_id, bi);
  } else {
    // failovered worker
    barrier_tokens_[task_id] = token;
    if (barrier_infos_.empty()) {
      failover_barrier_infos_[task_id] = bi;
    } else {
      if (barrier_infos_.begin()->second.barrier_id == barrier_id) {
        BarrierAddAndNotifyAll(task_id, bi);
      } else {
        cb(Status::Ok());
      }
    }
  }
}

void SchedulerImpl::BarrierAddAndNotifyAll(
    int task_id, const BarrierV2Info& bi) {
  barrier_infos_[task_id] = bi;
  if (barrier_infos_.size() == worker_count_ - finished_workers_.size()) {
    for (auto& it: barrier_infos_) {
      (it.second.cb)(Status::Ok());
    }

    barrier_infos_.clear();
  }
}

void SchedulerImpl::InternalGetWorkerFinishCount(Version version, function<void (int64_t, const Status&)> cb) {
  if (version != version_) {
    cb(-1, VersionMismatch(version_, version));
    return;
  }
  int64_t size = finished_workers_.size();
  cb(size, Status::Ok());
}

void SchedulerImpl::InternalSynchronizeEnter(Version version, int id, int worker_count,
        function<void (int64_t, const Status&)> cb) {
  if (version != version_) {
    cb(-1, VersionMismatch(version_, version));
    return;
  }
  if (!sync_) {
    auto sync = new Synchronizer(worker_count);
    sync_.reset(sync);
  }
  auto sync = dynamic_cast<Synchronizer*>(sync_.get());
  if (sync == nullptr) {
    LOG(ERROR) << "Call sync method in async mode.";
    cb(-1, Status::ArgumentError("Call sync method in async mode."));
  }
  sync->Enter(id, cb);    
}

void SchedulerImpl::InternalSynchronizeLeave(Version version, int id, int64_t token, function<void (const Status&)> cb) {
  if (version != version_) {
    cb(VersionMismatch(version_, version));
    return;
  }
  auto sync = dynamic_cast<Synchronizer*>(sync_.get());
  if (sync == nullptr) {
    LOG(ERROR) << "Call sync method in async mode.";
    cb(Status::ArgumentError("Call sync method in async mode."));
  }
  sync->Leave(id, token, cb);
}

Status SchedulerImpl::RestoreGlobalQueue(const std::string& ckpt_dir) {
  std::unique_ptr<FileSystem::ReadStream> s;
  Status st = FileSystem::OpenReadStreamAny(ckpt_dir + "/global_queue_meta", &s);    
  if (st.IsOk()) {
    std::string queue_state;
    PS_CHECK_STATUS(s->ReadStr(&queue_state));    
    PS_CHECK_STATUS(DeserializeGlobalQueue(queue_state));
  }

  return Status::Ok();
}

Status SchedulerImpl::SerializeGlobalQueue(std::string* buf) {
  std::unique_lock<std::mutex> lock(mu_);
  size_t size = global_file_queues_.size();
  buf->append((char*)&size, sizeof(size_t));
  for (auto& it: global_file_queues_) {
    size_t name_size = it.first.size();
    buf->append((char*)&name_size, sizeof(size_t));    
    buf->append((char*)it.first.data(), it.first.size());    
    std::string queue_state;
    it.second->Serialize(&queue_state);
    size_t queue_state_size = queue_state.size();
    buf->append((char*)&queue_state_size, sizeof(size_t));    
    buf->append((char*)queue_state.data(), queue_state.size());        
  }

  return Status::Ok();
}

Status SchedulerImpl::DeserializeGlobalQueue(const std::string& buf) {
  std::unordered_map<std::string, std::unique_ptr<GlobalFileQueue> > queues;
  char* ptr = const_cast<char*>(buf.data());
  size_t len = *(reinterpret_cast<size_t*>(ptr));
  ptr += sizeof(size_t);
  for (size_t i = 0; i < len; ++i) {
    size_t name_len = *(reinterpret_cast<size_t*>(ptr));    
    ptr += sizeof(size_t);
    std::string name;
    name.assign(ptr, name_len);
    ptr += name_len;
    size_t queue_state_len = *(reinterpret_cast<size_t*>(ptr));    
    ptr += sizeof(size_t);    
    std::string queue_state;
    queue_state.assign(ptr, queue_state_len);
    ptr += queue_state_len;    
    queues[name].reset(new GlobalFileQueue());
    PS_CHECK_STATUS(queues[name]->Deserialize(queue_state));
  }

  std::unique_lock<std::mutex> lock(mu_);
  global_file_queues_ = std::move(queues);
  return Status::Ok();
}

Status SchedulerImpl::ReadCheckpoints(const std::string& ckpt_dir, bool ignoreError, std::vector<std::string>* checkpoints) {
  std::unique_ptr<FileSystem::ReadStream> s;
  Status st = FileSystem::OpenReadStreamAny(ckpt_dir + "/checkpoints", &s);
  if (st.IsOk()) {
    size_t size;
    PS_CHECK_STATUS(s->ReadRaw(&size));
    checkpoints->resize(size);
    for (size_t i = 0; i < size; i++) {
      PS_CHECK_STATUS(s->ReadStr(&(*checkpoints)[i]));
    }
    return Status::Ok();
  } else if (ignoreError) {
    return Status::Ok();
  } else {
    return st;
  }
}

Status SchedulerImpl::ReadVariableInfoMeta(const std::string& path, size_t* server_num, std::vector<VariableInfo>* result) {
  std::unique_ptr<FileSystem::ReadStream> s;
  PS_CHECK_STATUS(FileSystem::OpenReadStreamAny(path + "/__meta__", &s));
  size_t infos_type;
  std::string infos_buf;
  PS_CHECK_STATUS(s->ReadRaw(server_num));
  PS_CHECK_STATUS(s->ReadRaw(&infos_type));
  PS_CHECK_STATUS(s->ReadStr(&infos_buf));
  Data* info_wrapper;
  size_t len;
  serializer::MemGuard mem;
  serializer::Fragment frag(&infos_buf[0], infos_buf.size());
  PS_CHECK_STATUS(serializer::DeserializeAny<Data>(infos_type, &frag, 0, &info_wrapper, &len, mem));
  std::unique_ptr<Data> info_wrapper_deleter(info_wrapper);
  WrapperData<VariableInfoCollection>* info_wrapper_converted = dynamic_cast<WrapperData<VariableInfoCollection>*>(info_wrapper);
  if (info_wrapper_converted == nullptr) {
    return Status::Unknown("Variable Info Load Error");
  }
  *result = info_wrapper_converted->Internal().infos;
  return Status::Ok();
}

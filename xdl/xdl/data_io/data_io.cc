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


#include "xdl/data_io/data_io.h"

#include "xdl/core/lib/timer.h"
#include "xdl/core/framework/cpu_device.h"
#ifdef USE_PS_PLUS
#include "xdl/data_io/global_scheduler.h"
#endif

#include "google/protobuf/text_format.h"

namespace xdl {
namespace io {

static const unsigned kTimeWait = 60000; /// millisecond
static const unsigned kTimeWaitTORetry = 100; /// millisecond

DataIO::DataIO(const std::string &ds_name, ParserType parser_type,
               FSType fs_type, const std::string &namenode,
               size_t worker_id, bool global_schedule)
    : ds_name_(ds_name), parser_type_(parser_type),
      fs_type_(fs_type) {
  fs_ = GetFileSystem(fs_type, namenode.empty()?nullptr:namenode.c_str());
#ifdef USE_PS_PLUS
  if (global_schedule) {
    sched_.reset(new GlobalScheduler(fs_, ds_name, 1, worker_id));
  }
#endif
  if (sched_ == nullptr) {
    sched_.reset(new Scheduler(fs_));
  }
  schema_.reset(new Schema);
  DataIOMap::Add(ds_name, this);
}

DataIO::~DataIO() {
  XDL_LOG(DEBUG) << ds_name_ << " destroyed!";
  Shutdown(true);
  DataIOMap::Delete(ds_name_);
}

bool DataIO::Init() {
  if (sgroup_q_ == nullptr) {
    sgroup_q_ = new BlockingQueue<SGroup *>(schema_->batch_size_*threads_read_);
  }

  
  if (batch_q_ == nullptr) {
    batch_q_ = new BlockingQueue<Batch *>(threads_);
  }

  if (meta_data_.empty() && !meta_path_.empty()) {
    meta_data_ = fs_->Read(meta_path_);
  }

  parsers_.clear();
  for (size_t i = 0; i < threads_read_; ++i) {
    Parser* parser = new Parser(parser_type_, schema_.get());
    XDL_CHECK(parser->InitMeta(meta_data_));
    parsers_.emplace_back(parser);
  }

  packers_.clear();
  for (size_t i = 0; i < threads_; ++i) {
    auto packer = new Packer(schema_.get(), new CpuDevice());
    packers_.emplace_back(packer);
  }

  mergers_.clear();
  for (size_t i = 0; unique_ && i < packers_.size(); ++i) {
    auto merger = new Merger(schema_.get(), new CpuDevice());
    mergers_.emplace_back(merger);
  }

  sched_->Schedule();

  curr_ = (Batch*)-1;
  next_ = (Batch*)-1;
  return true;
}

/// finish while rparam is nullptr, or shutdown
bool DataIO::DoParse(size_t tid) {
  XDL_LOG(DEBUG) << "parser." << tid << " startup";
  assert(tid < parsers_.size());
  auto parser = parsers_[tid].get();
  XDL_LOG(DEBUG) << "this=" << this << ", parser=" << parser;
  size_t count_rparam = 0;
  size_t count_sgroup = 0;
  while(running_) {
    ReadParam *rparam = sched_->Acquire();
    if (rparam == nullptr) {
      break;
    }
    ++count_rparam;

    parser->Init(rparam);
    while(running_) {
      auto sgroup = parser->Run();
      if (sgroup == END) {
        break;
      }
      XDL_CHECK(sgroup != nullptr);
      ++count_sgroup;

      std::unique_lock<std::mutex> lck(mutex_);
      while (pause_) {
        cv_.wait(lck);
      }
      while(!sgroup_q_->TryEnqueue(sgroup, kTimeWaitTORetry)) {
        if (!running_) { break; }
      } 

      if (++parse_count_ == parse_limit_) {
        pause_ = true;
        XDL_LOG(DEBUG) << "parse limit " << parse_limit_ << ", pause ...";
      }
    }
    sched_->Release(rparam);
  }

  XDL_LOG(DEBUG) << "parser." << tid << " shutdown rparam=" 
      << count_rparam << " sgroups=" << count_sgroup ;
}

/* 1. all parser done, dequeue END, quit
 * 2. exactly exhaust all sgroup, but parsers not done, push END to batch
 * 3. exactly exhaust all sgroup, and parsers done, push END to batch, quit
 * 4. dequeue sgroup timeout, push END to batch
 */
bool DataIO::DoPack(size_t tid) {
  XDL_LOG(DEBUG) << "packer." << tid << " startup";
  assert(tid < packers_.size());
  auto packer = packers_[tid].get();
  auto merger = unique_ ? mergers_[tid].get() : nullptr;
  //XDL_LOG(DEBUG) << "this=" << this << ", packer=" << packer;
  size_t count_sgroup = 0;
  size_t count_batch = 0;
  while(running_) {
    //XDL_TIMER_NOW(deque_sgroup);
   again:
    SGroup *sgroup = nullptr;
    if (!sgroup_q_->TryDequeue(&sgroup, kTimeWait)) {
      if (!wait_exactly_) {
        //continue;
        goto again;
      }
      // in pause mode
      sgroup = (SGroup *)END;
      XDL_LOG(DEBUG) << "wait exactly timeout, push END to packer";
    }
    XDL_CHECK(sgroup != nullptr);
    //XDL_TIMER_STOP(deque_sgroup);

    //XDL_TIMER_NOW(run_ops);
    if (sgroup != END) {
      ++count_sgroup;
      if (!RunOps(sgroup)) {
        SGroupPool::Get()->Release(sgroup);
        std::unique_lock<std::mutex> lck(mutex_);
        /// all parser wait at cv, until parse_count == 0
        if (--parse_count_ <= 0) {
          /// it's the last sgroup, other packer wait at sgroup_q_
          pause_ = false;
          cv_.notify_all();

          XDL_LOG(DEBUG) << "all re parsers done, notify packers exit ...";
          NotifyPacker();
        }
        XDL_LOG(DEBUG) << "sgroup="<< sgroup << " -parse_count=" << parse_count_;
        continue;
      } else {
        XDL_LOG(DEBUG) << "sgroup="<< sgroup << " parse_count=" << parse_count_;
      }
      if (sgroup->size_ != sgroup->Get()->labels_size()) {
        if (sgroup->Get()->labels_size() == 0) {
          // Ops del all
          SGroupPool::Get()->Release(sgroup);
          continue;
        }
        XDL_DLOG(DEBUG) << "rebuild sgroup " << sgroup->size_ 
            << " -> " << sgroup->Get()->labels_size();
        sgroup->Reset(sgroup->begin_);
      }
    }
    //XDL_TIMER_STOP(run_ops);

    //XDL_TIMER_NOW(run_pack);
    std::vector<Batch *> batchs = packer->Run(sgroup);
    for(auto &batch: batchs) {
      ++count_batch;
      if (unique_) {
        batch = merger->Run(batch);
      }
      while (!batch_q_->TryEnqueue(batch, kTimeWaitTORetry)) {
        if (!running_) { break; }
      }
    }
    //XDL_TIMER_STOP(run_pack);

    if (sgroup == END && parsers_done_ && (parse_count_ == 0 || !wait_exactly_)) {
      XDL_LOG(DEBUG) << "quit, wait_exactly=" << wait_exactly_ << " parse_count=" << parse_count_;
      break;
    }
  }

  XDL_LOG(DEBUG) << "packer." << tid << " shutdown sgroups="
      << count_sgroup << " batchs=" << count_batch;
}

bool DataIO::Startup() {
  XDL_CHECK(!running_);
  Init();
  XDL_CHECK(sgroup_q_->Size() == 0) << "sgroup_q_ is not empty before start";
  XDL_CHECK(batch_q_->Size() == 0) << "batch_q_ is not empty before start";
  running_ = true;

  for (size_t i = 0; i < parsers_.size(); ++i) {
    th_parsers_.push_back(std::thread([this, i](){this->DoParse(i);}));
  }

  for (size_t i = 0; i < packers_.size(); ++i) {
    th_packers_.push_back(std::thread([this, i](){this->DoPack(i);}));
  }

  XDL_LOG(DEBUG) << "xdl.data_io startup";

  /// wait background
  th_wait_ = std::thread([this](){this->Wait();});

  return true;
}

bool DataIO::Wait() {
  /// wait to done
  for (size_t i = 0; i < parsers_.size(); ++i) {
    th_parsers_[i].join();
  }

  parsers_done_ = true;

  /// all parsers done, notify packer while not keep sgroup
  if (!schema_->keep_sgroup_) {
    /// keep means more sg will be ReParse to sgroup_q_ latter 
    XDL_LOG(DEBUG) << "all parsers done, notify packers exit ...";
    NotifyPacker();
  }

  for (size_t i = 0; i < packers_.size(); ++i) {
    th_packers_[i].join();
  }

  packers_done_ = true;
  th_parsers_.clear();
  th_packers_.clear();

  /// all packers done, notify get_batch_op
  batch_q_->ForceEnqueue(nullptr);
  XDL_LOG(DEBUG) << "all packers done, notify graph exit ...";
  return true;
}

bool DataIO::Restart(size_t start) {
  XDL_LOG(DEBUG) << "restart data_io " << ds_name_ << ", start_time=" << start;
  Shutdown(true);
  SetStartTime(start);
  Startup();
  return true;
}

bool DataIO::Shutdown(bool force) {
  if (!running_) {
    /// never startup
    XDL_LOG(DEBUG) << "xdl.data_io shutdown shortly";
    return false;
  }
  if (force) {
    running_ = false;
    NotifyParser();
    NotifyPacker();
    batch_q_->ForceEnqueue(nullptr);
    cv_.notify_all();
  }

  th_wait_.join();
  sgroup_q_->ClearAndDelete([](SGroup* sg) {
      if (sg != nullptr && sg != END) {
        SGroupPool::Get()->Release(sg);
      }
  });
  batch_q_->ClearAndDelete([](Batch* batch) {
      if (batch != nullptr) {
        BatchPool::Get()->Release(batch);
      }
  });
  sched_->Clear();
  XDL_LOG(DEBUG) << "xdl.data_io shutdown";
  return true;
}

bool DataIO::AddOp(Operator *op) {
  XDL_CHECK(!running_);
  XDL_CHECK(op != nullptr);
  if (op->set_schema(schema_.get()) == false)  return false;
  ops_.push_back(op);
  return true;
}

bool DataIO::SetMeta(const std::string &path) {
  XDL_CHECK(!running_);
  meta_path_ = path;
  return false;
}

bool DataIO::SetMetaData(const std::string& data) {
  XDL_CHECK(!running_);
  meta_data_ = data;
  return true;
}

bool DataIO::AddPath(const std::string &path) {
  XDL_CHECK(!running_);
  sched_->AddPath(path);
  return true;
}

bool DataIO::SetEpochs(size_t epochs) {
  XDL_CHECK(!running_);
  sched_->SetEpochs(epochs);
  return true;
}

bool DataIO::SetShuffle(bool shuffle) {
  XDL_CHECK(!running_);
  sched_->SetShuffle(shuffle);
  return true;
}

bool DataIO::SetZType(ZType ztype) {
  XDL_CHECK(!running_);
  sched_->SetZType(ztype);
  return true;
}

bool DataIO::SetBatchSize(size_t batch_size) {
  XDL_CHECK(!running_);
  schema_->batch_size_ = batch_size;
  return true;
}

bool DataIO::SetLabelCount(size_t label_count) {
  XDL_CHECK(!running_);
  schema_->label_count_ = label_count;
  return true;
}

bool DataIO::SetSplitGroup(bool split) {
  XDL_CHECK(!running_);
  schema_->split_group_ = split;
  return true;
}

bool DataIO::SetKeepSGroup(bool keep) {
  XDL_CHECK(!running_);
  /// if keep, no END will be raised by end of read
  /// END should be raised by IOP returning false for each samplegroup
  schema_->keep_sgroup_ = keep;
  return true;
}

bool DataIO::SetKeepSKey(bool keep) {
  XDL_CHECK(!running_);
  schema_->keep_skey_ = keep;
  return true;
}

bool DataIO::SetUniqueIds(bool unique) {
  XDL_CHECK(!running_);
  unique_ = unique;
  return true;
}

bool DataIO::GetUniqueIds() const {
  return unique_;
}

bool DataIO::SetPadding(bool pad) {
  XDL_CHECK(!running_);
  schema_->padding_ = pad;
  return true;
}

bool DataIO::SetPause(size_t limit, bool wait_exactly) {
  std::unique_lock<std::mutex> lck(mutex_);
  /// wait exactly must set keep sgroup, then wait for op return false
  XDL_CHECK(schema_->keep_sgroup_ || !wait_exactly);
  parse_limit_ = limit;
  wait_exactly_ = wait_exactly;
  if (wait_exactly_ && !check_finish_delay_) {
    XDL_LOG(WARNING) << "wait exactly must finish delay, set it";
    check_finish_delay_ = true;
  }
  pause_ = false;
  return true;
}

bool DataIO::SetThreads(size_t threads,  size_t threads_read) {
  XDL_CHECK(!running_);
  XDL_CHECK(threads > 0 && threads <= 32 && threads_read > 0 && threads_read <= 64) 
      << "threads=" << threads << " threads_read=" << threads_read;
  threads_ = threads;
  threads_read_ = threads_read;
  return true;
}

bool DataIO::SetStartTime(size_t ts) {
  return true;
}

bool DataIO::SetEndTime(size_t ts) {
  return true;
}

size_t DataIO::GetLatestTime() {
  return 0;
}

size_t DataIO::GetReaderOffset() {
  return 0;
}

bool DataIO::SetDuration(size_t dur) {
  return true;
}

bool DataIO::IsStreaming() {
  return false;
}

bool DataIO::SetFinishDelay(bool delay) {
  XDL_CHECK(!running_);
  check_finish_delay_ = delay;
  if (wait_exactly_ && !check_finish_delay_) {
    XDL_LOG(FATAL) << "wait exactly must check finish delay";
  }
  return true;
}

const FeatureOption *DataIO::GetFeatureOpt(const std::string &name) {
  return schema_->Get(name);
}

bool DataIO::AddFeatureOpt(const std::string &name, FeatureType type, int table, 
                           int nvec, const std::string &mask, bool serialized,
                           int cutoff, const std::string &dsl) {
  FeatureOption *opt = new FeatureOption();
  opt->set_name(name);
  opt->set_type(type);
  opt->set_serialized(serialized);
  XDL_CHECK(type == kSparse || nvec > 0);
  if (table >= 0)  { opt->set_table(table); }
  if (nvec > 0) { opt->set_nvec(nvec); }
  if (!mask.empty()) { XDL_CHECK(mask.size() == nvec) << "mask size=" << mask.size();
    opt->set_mask(mask); }
  if (cutoff != 0) { opt->set_cutoff(cutoff); }
  if (!dsl.empty()) { opt->set_dsl(dsl); }
  schema_->Add(opt);
  return true;
}

const std::vector<std::string> &DataIO::sparse_list() const {
  return schema_->sparse_list();
}

const std::vector<std::string> &DataIO::dense_list() const {
  return schema_->dense_list();
}

size_t DataIO::ntable() const {
  return schema_->ntable();
}

const std::string DataIO::name() const {
  return ds_name_;
}

FileSystem &DataIO::fs() {
  return *fs_;
}

bool DataIO::RunOps(SGroup *sgroup) {
  assert(sgroup != nullptr && sgroup != END);
  if (!sgroup->own_) {
    /// only keep group will reach here
    assert(schema_->keep_sgroup_);
    return true;
  }
  for (auto &op : ops_) {
    if (!op->Run(sgroup->Get())) {
      return false;
    }
  }
  return true;
}

const Batch *DataIO::GetBatch() {
  if (curr_ == nullptr) {
    return nullptr;
  }

  /// release prev
  ReleaseBatch();

  if (next_ == (Batch *)-1) {
    /// first time or check_finish_delay
    next_ = batch_q_->Dequeue(); 
  }

  /// get curr from next
  curr_ = next_;
  next_ = (Batch *)-1;

  // get next while no check_finish_delay
  if (curr_!=nullptr && !check_finish_delay_) {
    next_ = batch_q_->Dequeue();
  }

  ++ count_;

  return curr_;
}

bool DataIO::ReleaseBatch() {
  XDL_CHECK(curr_ != nullptr);
  if (curr_ == (Batch *)-1) {
    // has been release
    return false;
  }
  ReParse(curr_);
  curr_->Reuse();
  curr_ = (Batch *)-1;
  return true;
}

const Batch *DataIO::GetBatchNext() {
  if (!check_finish_delay_) {
    XDL_LOG(WARNING) << "next batch has been retrivaled, while 'check_finish_delay' not set";
    XDL_CHECK(next_ != (Batch *)-1);
    return next_;
  }
  if (curr_ != nullptr) {
    next_ = batch_q_->Dequeue();
  }
  return next_;
}

/// make sure thread-safe by user
//const Batch *DataIO::GetBatch() {
//  if (curr_ != nullptr) {
//    ReParse(curr_);
//    curr_->Reuse();
//    curr_ = next_;
//  }
//
//  if (count_ == 0 || check_finish_delay_) {
//    /// first time or finish delay
//    curr_ = batch_q_->Dequeue(); 
//  }
//
//  // get next while no finish_delay
//  if (!check_finish_delay_ && curr_ != nullptr) {
//    next_ = batch_q_->Dequeue(); 
//  } else {
//    next_ = nullptr;
//  }
//  ++ count_;
//  //XDL_LOG(DEBUG) << "get " << count_ << "th batch, curr=" << curr_ << " next=" << next_;
//  return curr_;
//}

Batch *DataIO::CurrBatch() {
  return curr_;
}

bool DataIO::finished() const {
  if (!running_ || curr_ == nullptr) {
    XDL_LOG(DEBUG) << "check_finish by curr";
    return true;
  }
  if (check_finish_delay_ && curr_ != nullptr && next_ == nullptr) {
    XDL_LOG(DEBUG) << "check_finish by next";
    return true;
  }
  return false;
}

bool DataIO::ReParse(SGroup *sgroup) {
  XDL_LOG(DEBUG) << "reparse sgroup=" << sgroup;
  sgroup_q_->ForceEnqueue(sgroup);
  return true;
}

bool DataIO::ReParse(Batch *batch) {
  XDL_LOG(DEBUG) << "reparse batch=" << batch;
  if (!schema_->keep_sgroup_) {
    return false;
  }
  auto &sgroups = batch->sgroups(); 
  XDL_CHECK(sgroups.size() > 0);
  for (auto sgroup: sgroups) {
    ReParse(sgroup);
  }
  sgroups.clear();
  return true;
}

bool DataIO::NotifyParser() {
  for (size_t i = 0; i < parsers_.size(); ++i) {
    parsers_[i]->Shutdown();
  }
  return true;
}

bool DataIO::NotifyPacker() {
  for (size_t i = 0; i < packers_.size(); ++i) {
    sgroup_q_->ForceEnqueue((SGroup *)END);
  }
  return true;
}

std::string DataIO::Store() {
  DSState ds_state;
  ds_state.set_ds_name(ds_name_);
  sched_->Store(&ds_state);
  std::string text;
  if (state_as_text_) {
    google::protobuf::TextFormat::PrintToString(ds_state, &text);
  } else {
    ds_state.SerializeToString(&text);
  }
  return text;
}

bool DataIO::Restore(const std::string &text) {
  DSState ds_state;
  if (state_as_text_) {
    google::protobuf::TextFormat::ParseFromString(text, &ds_state);
  } else {
    ds_state.ParseFromString(text);
  }
  bool ret = sched_->Restore(ds_state);
  
  return ret;
}

void DataIO::Destroy() {
  Shutdown(true);
  DataIOMap::Delete(ds_name_);
}

}  // namespace io
}  // namespace xdl

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

#include "xdl/core/framework/cpu_device.h"

#include "google/protobuf/text_format.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

static const unsigned kTimeWait = 60000; /// millisecond

DataIO::DataIO(const std::string &ds_name, ParserType parser_type,
               FSType fs_type, const std::string &namenode)
    : ds_name_(ds_name), parser_type_(parser_type) {
  fs_ = GetFileSystem(fs_type, namenode.empty()?nullptr:namenode.c_str());
  sched_ = new Scheduler(fs_);
  schema_ = new Schema();
  DataIOMap::Add(ds_name, this);
}

DataIO::~DataIO() {
  Shutdown();
  DataIOMap::Delete(ds_name_);
}

bool DataIO::Init() {
  XDL_CHECK(parsers_.empty() && packers_.empty()) << "double init";
  sgroup_q_ = new BlockingQueue<SGroup *>(schema_->batch_size_);
  batch_q_ = new BlockingQueue<Batch *>(10);

  std::string meta = "";
  if (!meta_path_.empty()) {
    meta = fs_->Read(meta_path_);
  }
  for (size_t i = 0; i < threads_; ++i) {
    auto parser = new Parser(parser_type_, schema_);
    XDL_CHECK(parser->InitMeta(meta));
    parsers_.push_back(parser);
  }

  for (size_t i = 0; i < threads_; ++i) {
    auto packer = new Packer(schema_, new CpuDevice());
    packers_.push_back(packer);
  }

  for (size_t i = 0; unique_ && i < threads_; ++i) {
    auto merger = new Merger(schema_, new CpuDevice());
    mergers_.push_back(merger);
  }

  sched_->Schedule();
  return true;
}

/// finish while rparam is nullptr, or shutdown
bool DataIO::DoParse(size_t tid) {
  XDL_LOG(DEBUG) << "parser." << tid << " startup";
  assert(tid < threads_);
  auto parser = parsers_[tid];
  XDL_DLOG(DEBUG) << "this=" << this << ", parser=" << parser;
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
      while(!sgroup_q_->TryEnqueue(sgroup, kTimeWait)) {
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
  assert(tid < threads_);
  auto packer = packers_[tid];
  auto merger = unique_ ? mergers_[tid] : nullptr;
  //XDL_DLOG(DEBUG) << "this=" << this << ", packer=" << packer;
  size_t count_sgroup = 0;
  size_t count_batch = 0;
  while(running_) {
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
        XDL_DLOG(DEBUG) << "sgroup="<< sgroup << " -parse_count=" << parse_count_;
        continue;
      } else {
        XDL_DLOG(DEBUG) << "sgroup="<< sgroup << " parse_count=" << parse_count_;
      }
    }

    std::vector<Batch *> batchs = packer->Run(sgroup);
    for(auto &batch: batchs) {
      ++count_batch;
      if (unique_) {
        batch = merger->Run(batch);
      }
      while (!batch_q_->TryEnqueue(batch, kTimeWait)) {
        if (!running_) { break; }
      }
    }

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
  running_ = true;

  for (size_t i = 0; i < threads_; ++i) {
    th_parsers_.push_back(std::thread([this, i](){this->DoParse(i);}));
  }

  for (size_t i = 0; i < threads_; ++i) {
    th_packers_.push_back(std::thread([this, i](){this->DoPack(i);}));
  }

  XDL_LOG(DEBUG) << "xdl.data_io startup";

  /// wait background
  th_wait_ = std::thread([this](){this->Wait();});

  return true;
}

bool DataIO::Wait() {
  /// wait to done
  for (size_t i = 0; i < threads_; ++i) {
    th_parsers_[i].join();
  }

  parsers_done_ = true;

  /// all parsers done, notify packer while not keep sgroup
  if (!schema_->keep_sgroup_) {
    /// keep means more sg will be ReParse to sgroup_q_ latter 
    XDL_LOG(DEBUG) << "all parsers done, notify packers exit ...";
    NotifyPacker();
  }

  for (size_t i = 0; i < threads_; ++i) {
    th_packers_[i].join();
  }

  packers_done_ = true;

  /// all packers done, notify get_batch_op
  batch_q_->ForceEnqueue(nullptr);
  XDL_LOG(DEBUG) << "all packers done, notify graph exit ...";
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
  XDL_LOG(DEBUG) << "xdl.data_io shutdown";
  return true;
}

bool DataIO::AddOp(Operator *op) {
  XDL_CHECK(!running_);
  XDL_CHECK(op != nullptr);
  if (op->set_schema(schema_) == false)  return false;
  ops_.push_back(op);
  return true;
}

bool DataIO::SetMeta(const std::string &path) {
  XDL_CHECK(!running_);
  meta_path_ = path;
  return false;
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
  if (wait_exactly_ && !finish_delay_) {
    XDL_LOG(WARNING) << "wait exactly must finish delay, set it";
    finish_delay_ = true;
  }
  pause_ = false;
  return true;
}

bool DataIO::SetThreads(size_t threads) {
  XDL_CHECK(!running_);
  XDL_CHECK(threads > 0 && threads <= 32);
  threads_ = threads;
  return true;
}

bool DataIO::SetFinishDelay(bool delay) {
  XDL_CHECK(!running_);
  finish_delay_ = delay;
  if (wait_exactly_ && !finish_delay_) {
    XDL_LOG(FATAL) << "wait exactly must finish delay";
  }
  return true;
}

const FeatureOption *DataIO::GetFeatureOpt(const std::string &name) {
  return schema_->Get(name);
}

bool DataIO::AddFeatureOpt(const std::string &name, FeatureType type, int table, 
                           int nvec, bool serialized, const std::string &dsl) {
  FeatureOption *opt = new FeatureOption();
  opt->set_name(name);
  opt->set_type(type);
  opt->set_serialized(serialized);
  if (nvec > 0)  opt->set_nvec(nvec);
  if (table >= 0)  opt->set_table(table);
  if (!dsl.empty())  opt->set_dsl(dsl);
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
  if (sgroup->size_ != sgroup->Get()->labels_size()) {
    XDL_CHECK(!schema_->split_group_) 
        << "must set split group to false while rebuilding";
    XDL_DLOG(DEBUG) << "rebuild sgroup " << sgroup->size_ 
        << " -> " << sgroup->Get()->labels_size();
    sgroup->Reset(sgroup->begin_);
  }
  return true;
}

/// make sure thread-safe by user
const Batch *DataIO::GetBatch(unsigned msec) {
  if (curr_ != nullptr) {
    ReParse(curr_);
    curr_->Reuse();
    curr_ = next_;
  }

  if (count_ == 0 || finish_delay_) {
    /// first time or finish delay
    curr_ = batch_q_->Dequeue(); 
  }

  // get next while no finish_delay
  if (!finish_delay_ && curr_ != nullptr) {
    next_ = batch_q_->Dequeue(); 
  } else {
    next_ = nullptr;
  }
  ++ count_;
  XDL_LOG(DEBUG) << "get " << count_ << "th batch, curr=" << curr_ << " next=" << next_;
  return curr_;
}

Batch *DataIO::CurrBatch() {
  return curr_;
}

bool DataIO::finish() const {
  if (!running_ || curr_ == nullptr) {
    return true;
  }
  if (!finish_delay_ && curr_ != nullptr && next_ == nullptr) {
    return true;
  }
  return false;
}

bool DataIO::ReParse(SGroup *sgroup) {
  XDL_DLOG(DEBUG) << "reparse sgroup=" << sgroup;
  sgroup_q_->ForceEnqueue(sgroup);
  return true;
}

bool DataIO::ReParse(Batch *batch) {
  XDL_DLOG(DEBUG) << "reparse batch=" << batch;
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
  for (size_t i = 0; i < threads_; ++i) {
    parsers_[i]->Shutdown();
  }
  return true;
}

bool DataIO::NotifyPacker() {
  for (size_t i = 0; i < threads_; ++i) {
    sgroup_q_->ForceEnqueue((SGroup *)END);
  }
  return true;
}

std::string DataIO::Store() {
  DSState ds_state;
  ds_state.set_ds_name(ds_name_);
  sched_->Store(&ds_state);
  std::string text;
  google::protobuf::TextFormat::PrintToString(ds_state, &text);
  return text;
}

bool DataIO::Restore(const std::string &text) {
  if (running_) {
    XDL_LOG(DEBUG) << "runing without restore, only load parameters";
    return false;
  }
  DSState ds_state;
  google::protobuf::TextFormat::ParseFromString(text, &ds_state);
  XDL_CHECK(ds_name_ == ds_state.ds_name());

  return sched_->Restore(ds_state);
}


}  // namespace io
}  // namespace xdl

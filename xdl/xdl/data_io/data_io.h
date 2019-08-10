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

#ifndef XDL_IO_DATA_IO_H_
#define XDL_IO_DATA_IO_H_

#include <map>
#include <set>
#include <string>
#include <thread>

#include "xdl/core/lib/blocking_queue.h"
#include "xdl/data_io/constant.h"
#include "xdl/data_io/batch.h"
#include "xdl/data_io/fs/file_system.h"
#include "xdl/data_io/parser/parser.h"
#include "xdl/data_io/packer/packer.h"
#include "xdl/data_io/merger/merger.h"
#include "xdl/data_io/pool.h"
#include "xdl/data_io/op/op.h"
#include "xdl/data_io/scheduler.h"
#include "xdl/proto/sample.pb.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

class DataIO {
 public:
  DataIO() = delete;
  DataIO(const std::string &ds_name, ParserType parser_type=kPB,
         FSType fs_type=kLocal, const std::string &namenode="", 
         size_t worker_id=0, bool global_schedule=false);
  virtual ~DataIO();

  bool Init();
  /*!\brief destroy data io */
  void Destroy();
  /*!\brief start data io */
  bool Startup();
  /*!\brief restart data io from a timestamp */
  bool Restart(size_t start);
  /*!\brief stop data io */
  bool Shutdown(bool force=false);

  bool AddOp(Operator *op);

  /*!\brief add meta to read */
  bool SetMeta(const std::string &path);


  /*!\brief add meta data to read */
  bool SetMetaData(const std::string &data);

  /*!\brief add path to read */
  bool AddPath(const std::string &path);

  /*!\brief set epochs, 0 means loop forever */
  bool SetEpochs(size_t epochs);

  /*!\brief set shuffle */
  bool SetShuffle(bool shuffle);

  /*!\brief set batch size, 0 means variable size without padding */
  bool SetBatchSize(size_t batch_size=1024);

  /*!\brief set label count, default 2 */
  bool SetLabelCount(size_t label_count=2);

  /*!\brief set if could split sample group while batching, default true */
  bool SetSplitGroup(bool split=true);

  /*!\brief set if keep sgroup with batch, default false */
  bool SetKeepSGroup(bool keep=true);

  /*!\brief set if keep skey for debug, default false */
  bool SetKeepSKey(bool keep=true);

  /*!\brief set if unique key, default true */
  bool SetUniqueIds(bool unique=true);
  bool GetUniqueIds() const;

  /*!\brief set finish by next batch is null */
  bool SetFinishDelay(bool delay=true);

  /*!\brief set if padding to batch size, default true */
  bool SetPadding(bool pad=true);

  /*!\brief set compress format */
  bool SetZType(ZType ztype);

  /*!\brief set pause limit of sample, this will also unpause parser currently */
  bool SetPause(size_t limit, bool wait_exactly = false);

  /*!\brief set num of threads */
  bool SetThreads(size_t threads, size_t threads_read=8);

  /*!\brief set start point for online learning */
  bool SetStartTime(size_t ts);

  /*!\brief set end point for online learning */
  bool SetEndTime(size_t ts);

  /*!\brief get latest time of streaming reader */
  size_t GetLatestTime();

  /*!\brief get io offset of current reader */
  size_t GetReaderOffset();

  /*!\brief set time window for online learning */
  bool SetDuration(size_t dur);

  /*!\brief check if streaming io */
  bool IsStreaming();

  bool AddFeatureOpt(const std::string &name, FeatureType type, int table = 0,
                     int nvec = 0, const std::string &mask = "", bool serialized = false,
                     int cutoff=0, const std::string &dsl = "");

  const FeatureOption *GetFeatureOpt(const std::string &name);

  const std::vector<std::string> &sparse_list() const;
  const std::vector<std::string> &dense_list() const;
  size_t ntable() const;
  const std::string name() const;

  FileSystem &fs();

  bool RunOps(SGroup *sg);
  bool DoParse(size_t tid);
  bool DoPack(size_t tid);

  bool Wait();

  const Batch *GetBatch();
  const Batch *GetBatchNext();
  bool ReleaseBatch();
  Batch *CurrBatch();
  bool finished() const;

  bool ReParse(Batch *batch);
  bool ReParse(SGroup *sgroup);

  bool NotifyParser();
  bool NotifyPacker();

  std::string Store();
  bool Restore(const std::string &pbstr);

 protected:
  std::string ds_name_;
  ParserType parser_type_ = kPB;
  std::unique_ptr<Scheduler> sched_;
  std::unique_ptr<Schema> schema_;
  FSType fs_type_ = kLocal;
  FileSystem *fs_ = nullptr;

  size_t threads_ = 1;
  size_t threads_read_ = 8;
  bool unique_ = false;
  bool check_finish_delay_ = false;
  std::vector<Operator *> ops_;

  std::string meta_path_;
  std::string meta_data_;
  std::vector<std::unique_ptr<Parser>> parsers_;
  std::vector<std::unique_ptr<Packer>> packers_;
  std::vector<std::unique_ptr<Merger>> mergers_;

  std::vector<std::thread> th_parsers_;
  std::vector<std::thread> th_packers_;
  std::thread th_wait_;

  bool running_ = false;
  bool parsers_done_ = false;
  bool packers_done_ = false;

  BlockingQueue<SGroup*> *sgroup_q_ = nullptr;
  BlockingQueue<Batch*> *batch_q_ = nullptr;
  /// -1 : begin, nullptr: end
  Batch *curr_ = (Batch *)-1;
  Batch *next_ = (Batch *)-1;
  size_t count_ = 0;

  bool wait_exactly_ = false;
  bool state_as_text_ = false;

  std::mutex mutex_;
  std::condition_variable cv_;
  bool pause_ = false;
  size_t parse_count_ = 0;
  size_t parse_limit_ = ULONG_MAX;
};

class DataIOMap: public std::map<const std::string, DataIO *>, public Singleton<DataIOMap> {
 public:
  static DataIO *Get(const std::string &ds_name) {
    DataIOMap *data_io_map = DataIOMap::Instance();
    const auto &iter = data_io_map->find(ds_name);
    if (iter == data_io_map->end())  return nullptr;
    return iter->second;
  }

  static void Add(const std::string &ds_name, DataIO *data_io) {
    DataIOMap &data_io_map = *DataIOMap::Instance();
    XDL_CHECK(data_io_map[ds_name] == nullptr);
    data_io_map[ds_name] = data_io;
  }

  static void Delete(const std::string &ds_name) {
    XDL_LOG(DEBUG) << "delete data_io " << ds_name;
    DataIOMap &data_io_map = *DataIOMap::Instance();
    data_io_map.erase(ds_name);
  }
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_DATA_IO_H_

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

#ifndef PS_PLUS_CLIENT_MERGED_PROCESS_CONTEXT_H
#define PS_PLUS_CLIENT_MERGED_PROCESS_CONTEXT_H

#include <iostream>
#include <future>

namespace ps {
namespace client {

class MergedProcessContext {
 public:
  using Callback = std::function<void (const Status&)>;
  MergedProcessContext(size_t count) : count_(count), st_(Status::Ok()) {}
  void Collect(const std::vector<MergedPartitioner*>& combiner,
               MergedPartitionerContext* ctx,
               std::vector<Data*>* server_results,
               std::vector<std::vector<std::unique_ptr<Data> > >* results,
               size_t server_id,
               const Status& resp_st,
               const Callback& cb) {
    if (!resp_st.IsOk()) {
      st_ = resp_st;
    } else {
      if ((*server_results).size() != combiner.size()) {
        st_ = Status::ArgumentError("Combiner Size Error");
      } else {
        for (size_t i = 0; i < combiner.size(); ++i) {
          Status st = combiner[i]->Combine(ctx, (*server_results)[i], server_id, &(*results)[i]);
          if (!st.IsOk()) {
            st_ = st;
          }
        }
      }
    }
    delete server_results;
    if (--count_ == 0) {
      cb(st_);
      delete this;
    }
  }
  Callback CollectResults(const std::vector<MergedPartitioner*>& combiner,
                          MergedPartitionerContext* ctx,
                          std::vector<Data*>* server_results,
                          std::vector<std::vector<std::unique_ptr<Data> > >* results,
                          size_t server_id,
                          const Callback& cb) {
    return [combiner, ctx, server_results, results, server_id, cb, this](Status resp_st){
      this->Collect(combiner, ctx, server_results, results, server_id, resp_st, cb);
    };
  }
 private:
  std::atomic<size_t> count_;
  Status st_;
};

}  
}

#endif

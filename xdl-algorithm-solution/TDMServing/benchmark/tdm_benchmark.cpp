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

#include <pthread.h>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include "api/search_manager.h"
#include "util/str_util.h"
#include "util/log.h"
#include "util/timer.h"
#include "proto/search.pb.h"

namespace tdm_serving {

bool SampleToRequest(const std::string& sample,
                     SearchParam* req,
                     std::map<std::string, float>* gt_kv_map) {
  FeatureGroupList* user_feature =
      req->mutable_user_info()->mutable_user_feature();

  std::vector<std::string> split_1;
  util::StrUtil::Split(sample, '|', true, &split_1);
  if (split_1.size() < 3) {
    LOG_ERROR << "sample field size < 3";
    return false;
  }

  std::string sample_key = split_1[0];
  std::string sample_group = split_1[1];

  std::vector<std::string> split_2;
  util::StrUtil::Split(split_1[2], ';', true, &split_2);
  for (size_t i = 0; i < split_2.size(); i++) {
    std::vector<std::string> split_3;
    util::StrUtil::Split(split_2[i], '@', true, &split_3);
    if (split_3.size() < 2) {
      LOG_ERROR << "feature field size < 2";
      return false;
    }

    std::string feature_name = split_3[0];

    // add feature to req
    FeatureGroup* feature_group = NULL;
    if (feature_name != "test_unit_id") {
      feature_group = user_feature->add_feature_group();
      feature_group->set_feature_group_id(feature_name);
    }

    std::vector<std::string> split_4;
    util::StrUtil::Split(split_3[1], ',', true, &split_4);
    for (size_t j = 0; j < split_4.size(); j++) {
      std::vector<std::string> split_5;
      util::StrUtil::Split(split_4[j], ':', true, &split_5);

      uint64_t id;
      std::string key;
      float value = 0.0;

      if (split_5.size() == 2) {
        key = split_5[0];
        if (!util::StrUtil::StrConvert<uint64_t>(split_5[0].c_str(), &id)) {
          LOG_WARN << "id of feature " << split_5[1] << " is not number";
          return false;
        }
        if (!util::StrUtil::StrConvert<float>(split_5[1].c_str(), &value)) {
          LOG_WARN << "value of feature "
                        << split_5[1] << " is not float";
          return false;
        }
      } else if (split_5.size() == 1) {
        key = split_5[0];
        if (!util::StrUtil::StrConvert<uint64_t>(split_5[0].c_str(), &id)) {
          LOG_WARN << "id of feature " << split_5[1] << " is not number";
          return false;
        }
        value = 1.0;
      } else {
        LOG_ERROR << "feature_eneity field size is not 1 or 2";
        return false;
      }

      if (feature_name != "test_unit_id") {
        FeatureEntity* feature_entity = feature_group->add_feature_entity();
        feature_entity->set_id(id);
        feature_entity->set_value(value);
      } else { // ground truth data
        (*gt_kv_map)[key] = value;
      }
    }
  }

  return true;
}

bool MakeRequests(std::vector<SearchParam>* reqs, uint32_t topn) {
  std::string sample_file_path = "./bench_data/userbehavoir_test_sample.dat";
  std::ifstream sample_file_handler(sample_file_path.c_str());

  if (!sample_file_handler) {
    LOG_ERROR << "open " << sample_file_path << " failed";
    return false;
  }

  std::string line;

  while (std::getline(sample_file_handler, line)) {
    reqs->push_back(SearchParam());

    SearchParam& req = reqs->back();
    std::map<std::string, float> gt_kv_map;

    // MakeRequst
    if (!SampleToRequest(line, &req, &gt_kv_map)) {
      LOG_ERROR << "parse sample [" << line << "] to request failed";
      return false;
    }

    req.set_topn(topn);
    req.set_index_name("item_tree_index");
  }

  return true;
}

struct ThreadInfo {
  pthread_t thread_id;
  std::vector<SearchParam>* reqs;
  SearchManager* search_manager;
  uint32_t topn;
  uint32_t loop_num;
};

void* BenchThreadProc(void* data) {
  ThreadInfo* ti = reinterpret_cast<ThreadInfo*>(data);

  for (size_t i = 0; i < ti->loop_num; i++) {
    // Search
    SearchResult res;
    if (!ti->search_manager->Search(ti->reqs->at(i), &res)) {
      LOG_ERROR << "search failed";
      return NULL;
    }

    if (res.result_unit_size() > static_cast<int>(ti->topn)) {
      LOG_ERROR << "result size > topn";
      return NULL;
    }

    LOG_INFO << "loop count: " << i;
  }

  return NULL;
}

void Benchmark(uint32_t topn,
               uint32_t thread_num,
               uint32_t loop_num) {
  bool ret = false;

  // Init SearchManager
  SearchManager search_manager;
  ret = search_manager.Init("./bench_data/conf/index.conf",
                            "./bench_data/conf/model.conf");
  if (ret != true) {
    LOG_ERROR << "Init SearchManager failed";
    return;
  }

  // MakeSearchRequests
  std::vector<SearchParam> reqs;
  if (!MakeRequests(&reqs, topn)) {
    LOG_ERROR << "Make requests failed";
    return;
  }

  // Parallel Search
  std::vector<ThreadInfo> thread_infos;
  for (size_t i = 0; i < thread_num; i++) {
    thread_infos.push_back(ThreadInfo());
    ThreadInfo& thread_info = thread_infos.back();
    thread_info.reqs = &reqs;
    thread_info.search_manager = &search_manager;
    thread_info.topn = topn;
    thread_info.loop_num = loop_num;
  }

  util::Timer timer;
  timer.Start();

  for (size_t i = 0; i < thread_num; i++) {
    pthread_create(&thread_infos[i].thread_id, NULL,
                   &BenchThreadProc, &thread_infos[i]);
  }

  for (size_t i = 0; i < thread_num; i++) {
    pthread_join(thread_infos[i].thread_id, NULL);
  }

  timer.Stop();

  LOG_INFO << "Elapsed time=" << timer.GetElapsedTime();
  LOG_INFO << "Avg time=" << timer.GetElapsedTime() / loop_num;
  LOG_INFO << "query per second" << (loop_num * thread_num) / timer.GetElapsedTime();
}

}  // namespace tdm_serving

int main(int /*argc*/, char** /*argv*/) {
  LOG_CONFIG("tdm_benchmark", ".", 0);

  uint32_t search_topn = 200;
  uint32_t paralle_thread_num = 2;
  uint32_t loop_num_per_thread = 1000;

  tdm_serving::Benchmark(search_topn,
                         paralle_thread_num,
                         loop_num_per_thread);

  return 0;
}


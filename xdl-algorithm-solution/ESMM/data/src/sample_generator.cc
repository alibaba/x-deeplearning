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

/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <cstdlib>
#include <ctime>

#include "sample.pb.h"
#include "blocking_queue.h"
#include "string_utils.h"

namespace xdl {
typedef BlockingQueue<std::vector<std::string>*> SampleQueue;
typedef BlockingQueue<io::SampleGroup*> SampleGroupQueue;

std::string GetPartStr(const std::string& src, char sep, int index) {
  int count = 0;
  std::string::size_type begin = 0, end;
  while (count < index) {
    begin = src.find(sep, begin) + 1;
    ++count;
  }
  end = src.find(sep, begin);
  return src.substr(begin, end - begin);
}

void FeatureLineBuilder(io::FeatureLine* fl, const std::string& line) {
  auto features = StringUtils::split(GetPartStr(line, ',', 2), "");
  std::unordered_map<std::string, io::Feature*> field_maps;
  for (auto& feature : features) {
    io::Feature* fea = nullptr;
    std::string field = GetPartStr(feature, '', 0);
    if (field_maps.find(field) == field_maps.end()) {
      fea = fl->add_features();
      fea->set_type(io::kSparse);
      fea->set_name(field);
      field_maps[field] = fea;
    } else {
      fea = field_maps[field];
    }
    io::FeatureValue* fv = fea->add_values();
    std::string id_values = GetPartStr(feature, '', 1);
    fv->set_key(std::stoi(GetPartStr(id_values, '', 0)));
    fv->set_value(std::stof(GetPartStr(id_values, '', 1)));
  }
}

void SampleGroupBuilder(const std::vector<std::string>& features,
                        const std::unordered_map<std::string, int>& feature_maps,
                        SampleQueue* samples_queue,
                        SampleGroupQueue* sg_queue) {
  std::vector<std::string>* samples;
  while (true) {
    samples = samples_queue->Dequeue();
    if (samples == nullptr) {
      break;
    }
    std::string key = GetPartStr(samples->at(0), ',', 3);
    // generate sample group
    io::SampleGroup* sg = new io::SampleGroup;
    io::FeatureTable* ft0 = sg->add_feature_tables();
    io::FeatureTable* ft1 = sg->add_feature_tables();
    io::FeatureLine* aux_fl = ft1->add_feature_lines();
    FeatureLineBuilder(aux_fl, features.at(feature_maps.at(key)));

    for (size_t i = 0; i < samples->size(); ++i) {
      auto& sample = samples->at(i);
      sg->add_sample_ids(GetPartStr(sample, ',', 0));
      io::Label* label = sg->add_labels();
      float clk = stof(GetPartStr(sample, ',', 1));
      float cvs = stof(GetPartStr(sample, ',', 2));
      label->add_values(clk);
      label->add_values(cvs);
      io::FeatureLine* fl = ft0->add_feature_lines();
      FeatureLineBuilder(fl, GetPartStr(sample, ',', 5));
      fl->set_refer(0);
    }
    sg_queue->Enqueue(sg);
    delete samples;
  }
  sg_queue->Enqueue(nullptr);
  std::cout << "SampleGroupBuilder finished\n";
}

void FileWriter(SampleGroupQueue* sg_queue, int worker_num, const std::string& out_file, int file_cnt) {
  std::vector<std::unique_ptr<std::ofstream>> fouts;
  for (int i = 0; i < file_cnt; ++i) {
    fouts.emplace_back(new std::ofstream(out_file + "." + std::to_string(i),
                                         std::ios::out | std::ios::binary));
  }
  std::srand(std::time(nullptr));
  int count = 0;
  while (true) {
    if (count == worker_num) {
      break;
    }
    io::SampleGroup* sg = sg_queue->Dequeue();
    if (sg == nullptr) {
      ++count;
      continue;
    }
    int size = static_cast<int>(sg->ByteSize());
    int index = std::rand() % file_cnt;
    fouts[index]->write(reinterpret_cast<char*>(&size), sizeof(int));
    sg->SerializeToOstream(fouts[index].get());
    delete sg;
  }
  std::cout << "FileWriter finished\n";
}
}  // namespace xdl


int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " skeleton_file common_feature_file output_prefix worker_num file_num\n";
    return -1;
  }
  int file_cnt = 1;
  if (argc == 6) {
    file_cnt = std::stoi(argv[5]);
  }
  std::ifstream fs(argv[1], std::ios::in);
  std::ifstream fc(argv[2], std::ios::in);
  int worker_num = std::stoi(argv[4]);
  std::string line;
  std::vector<std::string> features;
  std::unordered_map<std::string, int> feature_maps;
  int index = 0;
  while (std::getline(fc, line)) {
    feature_maps[xdl::GetPartStr(line, ',', 0)] = index++;
    features.push_back(std::move(line));
  }
  xdl::SampleQueue samples_queue;
  xdl::SampleGroupQueue sg_queue;
  std::vector<std::thread> workers;
  for (int i = 0; i < worker_num; ++i) {
    workers.emplace_back(xdl::SampleGroupBuilder, features, feature_maps, &samples_queue, &sg_queue);
  }

  std::thread writer(xdl::FileWriter, &sg_queue, worker_num, argv[3], file_cnt);

  std::string last_key = "";
  std::vector<std::string> samples;

  while(std::getline(fs, line)) {
    std::string key = xdl::GetPartStr(line, ',', 3);
    if (key != last_key) {
      if (!samples.empty()) {
        std::vector<std::string>* data = new std::vector<std::string>(std::move(samples));
        samples_queue.Enqueue(data);
      }
      samples.clear();
    }
    samples.push_back(std::move(line));
    last_key = key;
  }
  if (!samples.empty()) {
    std::vector<std::string>* data = new std::vector<std::string>(std::move(samples));
    samples_queue.Enqueue(data);
  }

  for (int i = 0; i < worker_num; ++i) {
    samples_queue.Enqueue(nullptr);
  }
  for (auto& worker : workers) {
    worker.join();
  }
  writer.join();
  return 0;
}

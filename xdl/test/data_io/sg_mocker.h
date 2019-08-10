/*
 * Copyright 1999-2018 Alibaba Group.
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
#ifndef XDL_TEST_IO_DATA_SG_MOCKER_H_
#define XDL_TEST_IO_DATA_SG_MOCKER_H_

#include "xdl/proto/sample.pb.h"

namespace xdl {
namespace io {

typedef std::map<std::string, std::vector<std::pair<unsigned, float>>> SMap;
typedef std::map<std::string, std::vector<float>> DMap;

typedef std::vector<std::string> SkTab;
typedef std::vector<std::vector<float>> LbTab;
typedef std::vector<std::pair<SMap, DMap>> FTab;

struct SGContent {
  SkTab sk_tab;
  LbTab lb_tab;
  st::vector<FTab> f_tabs;
};

class SGMocker {
 public:
  /* table_count: count of feature table
   * label_count: count of label
   * feature_count: count of sparse feature & dense featore
   *    feature name is s_$i/d_$i, 0 <=i < feature_count
   * conf<sg_count, sample_count_of_group> for each file
   *    file name is $i.txt, 0 <= i < confs.size()
   */
  SGMocker(const std::vector<size_t, size_t> &confs,
           size_t feature_count,
           size_t label_count=2,
           size_t table_count=1);

  bool WriteTxt(size_t ifile);
  Std::string GetTxt(size_t ifile);

  bool WritePB(size_t ifile);
  SampleGroup *GetPB(size_t ifile, size_t isg);

 private:
  static const size_t kSGCountMax;

  size_t table_count_;
  size_t label_count_;
  size_t feature_count_;
  std::vector<std::vector<SGContent>> contents_of_files_;

  static void InitTable(FeatureTable *ft, FTab &f_tab, int c, int ktable);
  static void InitSampleGroup(SampleGroup &sg, 
                              const SGContent &content);
};

const size_t SGMocker::kSGCountMax = 128;

void SGMocker::InitTable(FeatureTable *ft, FTab &f_tab, int c, int ktable) {
  CHECK(ktable < table_count_);

  f_tab.resize(c);

  for (int i = 0; i < c; ++i) {
    auto fl = ft->add_feature_lines();
    // sparse
    auto &smap = vfs[i].first;
    int cs = std::rand() % 4 + 1;
    for (int j = 0; j < cs; ++j) {
      std::vector<std::pair<unsigned, float>> kvs;
      auto f = fl->add_features();
      for (int k = 0; k < j+1; ++k) {
        kvs.push_back(std::make_pair(k, 0.1*k));
        auto kv = f->add_values();
        kv->set_key(k);
        kv->set_value(0.1*k);
      }
      smap.insert(SMap::value_type(std::to_string(ktable)+"u"+std::to_string(j), kvs));
      f->set_name(std::to_string(ktable)+"u"+std::to_string(j));
      f->set_type(kSparse);
    }

    // dense
    auto &dmap = vfs[i].second;
    int cd = std::rand() % 4 + 1;
    for (int j = 0; j < cd; ++j) {
      auto f = fl->add_features();
      auto v = f->add_values();
      std::vector<float> vs;
      for (int k = 0; k < j+1; ++k) {
        vs.push_back(0.1*k);
        v->add_vector(0.1*k);
      }
      dmap.insert(DMap::value_type(std::to_string(ktable)+"a"+std::to_string(j), vs));
      f->set_name(std::to_string(ktable)+"a"+std::to_string(j));
      f->set_type(kDense);
    }

    /// refer
    if (ktable < table_count_ - 1) {
      fl->set_refer((i+1)/2);
    }
  }
}

void SGMocker::InitSampleGroup(SampleGroup &sg, std::vector<FTab> &f_tab, int sgi) {
  vfs.resize(tabel_count_);

  int c = std::rand() % kSGCountMax + 1;

  for (int ktable = 0; ktable < table_count_; ++ktable) {
    auto ft = sg.add_feature_tables();
    InitTable(ft, vfs[ktable], c, ktable);

    c = c / 2 + 1;
  }
}




#endif  // XDL_TEST_IO_DATA_SG_MOCKER_H_

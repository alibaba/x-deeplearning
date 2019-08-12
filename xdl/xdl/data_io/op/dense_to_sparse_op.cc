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

#include "xdl/data_io/op/dense_to_sparse_op.h"

#include <iostream>

namespace xdl {
namespace io {

bool DenseToSparseOP::Init(const std::map<std::string, std::string> &params) {
  for (auto kv: params) {
    std::vector<float> bounds;
    std::cout << "init: " << kv.first << "->" << kv.second << std::endl;
    auto &stags = kv.second;
    std::string::size_type end, beg = 0;
    do {
      end = stags.find(',', beg);
      bounds.push_back(atof(stags.substr(beg, end-beg).c_str()));
      beg = end + 1;
    } while (end != std::string::npos);
    boundaries_[kv.first]=bounds;

  }
  return true;
}

bool DenseToSparseOP::Run(SampleGroup *sg) {
  if (boundaries_.empty()) {
    return true;
  }

  auto ft = sg->mutable_feature_tables(0);
  int count = ft->feature_lines_size();
  for(int i = 0; i < count; ++i){
      auto fl = ft->mutable_feature_lines(i);
	  int f_count = fl->features_size();
	  for(int m = 0; m < f_count; m++){
		  auto feature = fl->mutable_features(m);
          if(feature->has_name()){
             auto itt = boundaries_.find(*(feature->mutable_name()));
             if (itt != boundaries_.end()){
                 auto f = fl->add_features();
                 f->set_name(itt->first);
                 f->set_type(kSparse);
                 auto float_values = feature->values(0).vector();
                 for(auto value : float_values){
                      auto pos = std::upper_bound(itt->second.begin(), itt->second.end(), value);
                      auto fv = f->add_values();
                      fv->set_key(pos - itt->second.begin());
                 }
			     feature->set_name(itt->first + "Dtos");
             }
          }
      }
  }

  return true;
}

XDL_REGISTER_IOP(DenseToSparseOP)

}  // namespace io
}  // namespace xdl


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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#include "tdm/selector.h"

#include "tdm/dist_tree.h"
#include "tdm/tree.pb.h"

namespace tdm {

class BrotherSelector: public Selector {
 public:
  bool Init(const std::string& config) override;
  void Select(const std::vector<int64_t>& input_ids,
              const std::vector<std::vector<int64_t> >& features,
              const std::vector<int>& layer_counts,
              int64_t* output_ids, float* weights) override;
};

bool BrotherSelector::Init(const std::string& config) {
  (void) config;
  return true;
}

void BrotherSelector::Select(const std::vector<int64_t>& input_ids,
                             const std::vector<std::vector<int64_t>>&,
                             const std::vector<int>& layer_counts,
                             int64_t* output_ids, float* weights) {
  auto nodes = dist_tree_->NodeById(input_ids);
  auto ancestors = dist_tree_->Ancestors(nodes);

  int layer_sum = layer_counts.size();
  for (auto it = layer_counts.begin(); it != layer_counts.end(); ++it) {
    layer_sum += *it;
  }

  memset(output_ids, 0x00, sizeof(int64_t) * layer_sum * input_ids.size());
  memset(weights, 0x00, sizeof(float) * layer_sum * input_ids.size());

  int i = 0;
  for (auto it = ancestors.begin(); it != ancestors.end(); ++it) {
    auto& ancs = *it;
    if (!ancs.empty()) {
      if (ancs.size() > layer_counts.size()) {
        ancs.resize(layer_counts.size());
      }

      int64_t* ids = output_ids + i * layer_sum;
      float* w = weights + i * layer_sum;
      auto silbings = dist_tree_->Silbings(ancs);
      for (size_t j = 0; j < ancs.size(); ++j) {
        // sample: +
        Node node;
        if (!node.ParseFromString(ancs[j].value)) {
          continue;
        }
        ids[0] = node.id();
        w[0] = 1;

        ++ids;
        ++w;

        // sample: -
        auto& sibs = silbings[j];
        for (int k = 0;
             k < layer_counts[j] && k < static_cast<int>(sibs.size()); ++k) {
          if (node.ParseFromString(sibs[k].value)) {
            ids[k] = node.id();
            w[k] = 0;
          }
        }

        ids += layer_counts[j];
        w += layer_counts[j];
      }
    }
    ++i;
  }
}

REGISTER_SELECTOR("by_brother", BrotherSelector);

}  // namespace tdm

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

#include <vector>
#include <thread>

#include "tdm/selector.h"
#include "tdm/dist_tree.h"
#include "tdm/tree.pb.h"
#include "tdm/store.h"
#include "tdm/common.h"

namespace tdm {

class LayerWiseSelector: public Selector {
 public:
  LayerWiseSelector();

  bool Init(const std::string& config) override;

  void Select(const std::vector<int64_t>& input_ids,
              const std::vector<std::vector<int64_t> >& features,
              const std::vector<int>& layer_counts,
              int64_t* output_ids, float* weights) override;

 private:
  bool with_prob_;
  std::vector<std::discrete_distribution<int> > node_prob_data_;
  int start_sample_layer_;
};

LayerWiseSelector::LayerWiseSelector(): with_prob_(false), 
	start_sample_layer_(-1) {
}

bool LayerWiseSelector::Init(const std::string& config) {
  auto conf = ParseConfig(config);
  if (conf.find("with_prob") != conf.end() && conf["with_prob"] == "true") {
    printf("[INFO] sample with prob\n");
    with_prob_ = true;
  }
   
  if (conf.find("start_sample_layer") != conf.end()) {
    start_sample_layer_ = atoi(conf["start_sample_layer"].c_str());
  }
  printf("[INFO] start_sample_layer %d\n", start_sample_layer_);
 
  for (int level = 0; level < dist_tree_->max_level(); ++level) {
    auto level_itr = dist_tree_->LevelIterator(level);
    auto level_end = dist_tree_->LevelEnd(level);
    size_t level_node_num_start = static_cast<size_t>(
		pow(dist_tree_->branch(), level)) - 1;

    std::vector<float> aux_vec;
    do {
      Node node;
      if (node.ParseFromString(level_itr->value)) {
        //计算当前节点的偏移量
        size_t key_no = dist_tree_->KeyNo(level_itr->key); 
        size_t level_offset = key_no - level_node_num_start;
		if (level != dist_tree_->max_level() - 1) {
			assert(level_offset == aux_vec.size());
		}
        //中间的不存在的空节点填充概率为0
        while(aux_vec.size() < level_offset) {
	      aux_vec.push_back(0);	
		}
        aux_vec.push_back(node.probality());
      }
      ++level_itr;
    } while (level_itr != level_end);
    std::discrete_distribution<int> dd(aux_vec.begin(), aux_vec.end());
    node_prob_data_.push_back(dd);
  }

  return true;
}

void LayerWiseSelector::Select(
    const std::vector<int64_t>& input_ids,
    const std::vector<std::vector<int64_t>>& features,
    const std::vector<int>& layer_counts,
    int64_t* output_ids, float* weights) {
  (void) features;
  auto nodes = dist_tree_->NodeById(input_ids);
  auto ancestors = dist_tree_->Ancestors(nodes);
  
  // Sample sum(layer_counts) negative samples
  // and layer_counts.size() positive samples
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

      int level = dist_tree_->max_level();
      for (size_t j = 0; j < ancs.size() 
				&& level - 1 >= start_sample_layer_; ++j) {
        --level;  // Upward

        // sample: +
        Node node;
        if (!node.ParseFromString(ancs[j].value)) {
          continue;
        }

        int64_t positive_sample_id = node.id();
        ids[0] = node.id();
        w[0] = 1;
        ++ids;
        ++w;

        // sample: -
        std::unordered_set<int> neighbor_indices_set;
        std::vector<int> neighbor_indices;
        size_t cur_layer_count = layer_counts.at(level);
        neighbor_indices_set.reserve(cur_layer_count);
        neighbor_indices.reserve(cur_layer_count);

        size_t key_no = dist_tree_->KeyNo(ancs[j].key);

        size_t neighbors_count = static_cast<size_t>(
            pow(dist_tree_->branch(), level)) - 1;
        size_t level_node_num_start = static_cast<size_t>(
            pow(dist_tree_->branch(), level)) - 1;

        static __thread std::hash<std::thread::id> hasher;
        static __thread std::mt19937 rng(
            clock() + hasher(std::this_thread::get_id()));
        std::uniform_int_distribution<int> distrib(0, neighbors_count);
         
        if (false == with_prob_) {
          while (neighbor_indices_set.size() < cur_layer_count) {
            int q = 0;
            if (with_prob_) {
              q = node_prob_data_.at(level)(rng);
            } else {
              q = distrib(rng);
            }

            if (neighbor_indices_set.find(q) != neighbor_indices_set.end()) {
              continue;
            }

            // 判定节点是否在tree中存在
            auto rand_key_no = level_node_num_start + q;
            if (!dist_tree_->IsFiltered(rand_key_no)
                && key_no != rand_key_no) {
              neighbor_indices_set.insert(q);
              neighbor_indices.push_back(q);
            }
          }
        } else {
           while (neighbor_indices.size() < cur_layer_count) {
            int q = node_prob_data_.at(level)(rng);

            // 判定节点是否在tree中存在
            auto rand_key_no = level_node_num_start + q;
            if (!dist_tree_->IsFiltered(rand_key_no)) {
              neighbor_indices.push_back(q);
            }
          } 
        }

        auto negative_samples =
            dist_tree_->SelectNeighbors(ancs[j], neighbor_indices);
        assert(cur_layer_count == negative_samples.size());

        for (int k = 0; k < cur_layer_count; ++k) {
          if (node.ParseFromString(negative_samples[k].value)) {
            if (false == with_prob_) {
              assert(positive_sample_id != node.id());
            }
            ids[k] = node.id();
            w[k] = 0;
          }
        }

        ids += cur_layer_count;
        w += cur_layer_count;
      }
    }
    ++i;
  }
}

REGISTER_SELECTOR("by_layerwise", LayerWiseSelector);

}  // namespace tdm

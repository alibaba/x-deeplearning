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

#ifndef TDM_SELECTOR_H_
#define TDM_SELECTOR_H_

#include <stdint.h>

#include <string>
#include <vector>
#include <map>

namespace tdm {

class DistTree;

class Selector {
 public:
  Selector();
  virtual ~Selector();

  DistTree* dist_tree() const;
  void set_dist_tree(DistTree* dist_tree);

  virtual bool Init(const std::string& config) = 0;

  /**
   * Select samples from some layers
   *
   * @input_ids: leaf item ids
   * @layer_counts: layer select counts
   * @output_ids: select result item ids,
         Note: output_ids has been allocated enough space
   * @weights: weights of select item, Note space has been allocated
   */
  virtual void Select(const std::vector<int64_t>& input_ids,
                      const std::vector<std::vector<int64_t> >& features,
                      const std::vector<int>& layer_counts,
                      int64_t* output_ids, float* weights) = 0;

 protected:
  DistTree* dist_tree_;
};

class SelectorMapper {
 public:
  SelectorMapper(const std::string& name, Selector* selector);
  ~SelectorMapper();

  static Selector* GetSelector(const std::string& name);

  static bool LoadSelector(const std::string& so_path);

 private:
  Selector* selecor_;
  static std::map<std::string, Selector*> selector_map_;
};

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define REGISTER_SELECTOR(name, cls) \
  SelectorMapper CONCAT(selmap_, cls)(name, new cls);

}  // namespace tdm

#endif  // TDM_SELECTOR_H_

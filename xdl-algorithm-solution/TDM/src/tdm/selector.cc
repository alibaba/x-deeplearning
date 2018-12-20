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

#include <dlfcn.h>

#include <string>

namespace tdm {

Selector::Selector(): dist_tree_(NULL) {
}

Selector::~Selector() {
}

DistTree* Selector::dist_tree() const {
  return dist_tree_;
}

void Selector::set_dist_tree(DistTree* dist_tree) {
  dist_tree_ = dist_tree;
}

std::map<std::string, Selector*> SelectorMapper::selector_map_;

SelectorMapper::SelectorMapper(const std::string& name,
                               Selector* selector): selecor_(selector) {
  selector_map_.insert({name, selector});
}

SelectorMapper::~SelectorMapper() {
  delete selecor_;
}

Selector* SelectorMapper::GetSelector(const std::string& name) {
  auto it = selector_map_.find(name);
  if (it != selector_map_.end()) {
    return it->second;
  }
  return NULL;
}

bool SelectorMapper::LoadSelector(const std::string& so_path) {
  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle != NULL) {
    return true;
  }
  return false;
}

}  // namespace tdm

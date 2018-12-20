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

#include "tdm/dist_tree.h"

#include "include/api.h"

tree_handler API(new)() {
  return &tdm::DistTree::GetInstance();
}

void API(set_prefix)(tree_handler handler, const std::string& prefix) {
  if (handler == NULL) {
    return;
  }
  reinterpret_cast<tdm::DistTree*>(handler)->set_key_prefix(prefix);
}

void API(set_store)(tree_handler handler, void* store) {
  if (handler == NULL) {
    return;
  }
  reinterpret_cast<tdm::DistTree*>(handler)->set_store(
      reinterpret_cast<tdm::Store*>(store));
}

void API(set_branch)(tree_handler handler, int branch) {
  if (handler == NULL) {
    return;
  }
  reinterpret_cast<tdm::DistTree*>(handler)->set_branch(branch);
}

bool API(load)(tree_handler handler) {
  if (handler == NULL) {
    return false;
  }

  return reinterpret_cast<tdm::DistTree*>(handler)->Load();
}

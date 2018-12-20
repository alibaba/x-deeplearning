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

#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "tdm/local_store.h"
#include "tdm/tree.pb.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "%s <tree_data> <node_id>\n", argv[0]);
    return 1;
  }

  std::string tree_data = argv[1];
  int node_id  = atoi(argv[2]);
  auto store = tdm::Store::NewStore("");
  store->LoadData(tree_data);

  auto& tree = tdm::DistTree::GetInstance();
  tree.set_store(store);
  if (!tree.Load()) {
    fprintf(stderr, "Load tree failed!\n");
    return 3;
  }

  auto tree_node = tree.NodeById(node_id);
  if (!tree_node.valid()) {
    fprintf(stderr, "Node %d not exists in tree!\n", node_id);
    return 4;
  }

  printf("Node key length: %lu, value length: %lu\n",
         tree_node.key.length(), tree_node.value.length());

  tdm::Node node;
  if (!node.ParseFromString(tree_node.value)) {
    fprintf(stderr, "Invalid tree node get!\n");
    return 5;
  }

  printf("Got Node:\n %s\n", node.DebugString().c_str());
  return 0;
}

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

#ifndef TDM_SERVING_INDEX_TREE_TREE_H_
#define TDM_SERVING_INDEX_TREE_TREE_H_

#include <fcntl.h>
#include <unordered_map>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/coded_stream.h"
#include "common/common_def.h"
#include "index/tree/node.h"
#include "index/tree/node_map.h"
#include "index/tree/tree_meta.h"
#include "index/index_conf.h"
#include "util/str_util.h"
#include "util/conf_parser.h"
#include "util/concurrency/mutex.h"
#include "util/log.h"
#include "proto/tree.pb.h"

namespace tdm_serving {

class TreeIndexConf;

class Tree {
 public:
  Tree();
  ~Tree();

  bool Init(const TreeIndexConf* index_conf);

  Node* root() {
    return node_by_seq(0);
  }

  Node* node_by_seq(uint32_t seq) {
    if (seq >= tree_meta_.total_node_num_) {
      LOG_ERROR << "seq: " << seq << " exceeds total node num: "
                     << tree_meta_.total_node_num_;
      return NULL;
    }
    return nodes_ + seq;
  }

  Node* node_by_id(uint64_t node_id) {
    return node_maps_->GetNode(node_id);
  }

  // get tree total node num
  uint32_t total_node_num() {
    return tree_meta_.total_node_num_;
  }

  // get tree max level
  uint32_t max_level() {
    return tree_meta_.max_level_;
  }

 private:
  // tree index conf
  const TreeIndexConf* index_conf_;

  // tree data info
  Node* nodes_;

  // tree meta info
  TreeMeta tree_meta_;

  // node id map
  typedef std::unordered_map<uint64_t, Node*> NodeIdMap;
  NodeMaps* node_maps_;

  // pb info
  UIMeta pb_meta_;
  std::vector<UITree*> pb_trees_;

  // mutex
  util::SimpleMutex mutex_;

  DISALLOW_COPY_AND_ASSIGN(Tree);
};

template<typename Pb>
bool ParsePbFromBinaryFile(const std::string& file, Pb* pb) {
  if (pb == NULL) {
    LOG_ERROR << "Input pb is NULL";
    return false;
  }
  int fd = open(file.c_str(), O_RDONLY);
  if (fd == -1) {
    LOG_ERROR << "Open " << file << " failed";
    return false;
  }
  google::protobuf::io::ZeroCopyInputStream* raw_input
      = new google::protobuf::io::FileInputStream(fd);
  google::protobuf::io::CodedInputStream* coded_input
      = new google::protobuf::io::CodedInputStream(raw_input);

  coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);

  if (!pb->ParseFromCodedStream(coded_input)) {
    LOG_ERROR << "Parse pb from " << file << " failed";
    return false;
  }

  delete coded_input;
  delete raw_input;
  return true;
}

template<typename Pb>
bool ConvertPbToBinaryFile(const std::string& file, const Pb& pb) {
  int fd = open(file.c_str(), O_RDWR|O_CREAT|O_TRUNC, S_IRWXU);
  if (fd == -1) {
    LOG_ERROR << "open " << file << " failed";
    return false;
  }
  google::protobuf::io::ZeroCopyOutputStream* raw_output
      = new google::protobuf::io::FileOutputStream(fd);
  google::protobuf::io::CodedOutputStream* coded_output
      = new google::protobuf::io::CodedOutputStream(raw_output);

  if (!pb.SerializeToCodedStream(coded_output)) {
    LOG_ERROR << "serialize pb to " << file << " failed";
    return false;
  }

  delete coded_output;
  delete raw_output;
  return true;
}

}  // namespace tdm_serving

#endif  // TDM_SERVING_INDEX_TREE_TREE_H_

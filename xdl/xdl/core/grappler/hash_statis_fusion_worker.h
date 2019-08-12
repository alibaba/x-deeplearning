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

#ifndef XDL_CORE_GRAPPLER_HASH_STATIS_FUSION_WORKER
#define XDL_CORE_GRAPPLER_HASH_STATIS_FUSION_WORKER

#include "xdl/core/framework/grappler.h"
#include "xdl/core/grappler/ps_fusion_worker.h"

#include <atomic>

namespace xdl {

class HashStatisFusionWorker: public FusionWorker {
 public:
  HashStatisFusionWorker(const std::string& statis_type)
    : statis_type_(statis_type) {}
  Status Process(GraphDef* graph, OutputSpec* output) override;

 private:
  std::string statis_type_;

  bool NodeMatcher(NodeDef* n) {
    std::string var_type;
    if (!GetAttrValue(n, "var_type", &var_type).IsOk()) {
      return false;
    }
    std::string statis_type;
    if (!GetAttrValue(n, "statis_type", &statis_type).IsOk()) {
      return false;
    }
      
    return n->op == "PsSparseStatisOp" && statis_type_ == statis_type
      && (var_type == "hash128" || var_type == "hash64" || var_type == "hash");
  }

  Status PostCluster(
      const std::vector<std::set<NodeDef*> >& clusters,
      std::vector<std::set<NodeDef*> >* sub_clusters);
  Status DoFusion(
      const std::vector<std::set<NodeDef*> >& clusters);
  Status FuseOneCluster(
      const std::set<NodeDef*>& clusters);
  Status FuseImpl(
      const std::string& var_name_str,
      DataType itype,
      DataType otype,
      const std::set<NodeDef*>& cluster,
      NodeDef* fused_node);
};

}

#endif

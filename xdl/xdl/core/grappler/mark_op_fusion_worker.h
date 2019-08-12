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

#ifndef XDL_CORE_GRAPPLER_MARK_OP_FUSION_WORKER
#define XDL_CORE_GRAPPLER_MARK_OP_FUSION_WORKER

#include "xdl/core/framework/grappler.h"
#include "xdl/core/grappler/ps_fusion_worker.h"

#include <atomic>

namespace xdl {

class MarkOpFusionWorker: public FusionWorker {
 public:
  Status Process(GraphDef* graph, OutputSpec* output) override;

 private:
  bool NodeMatcher(NodeDef* n) {
    return n->op == "PsMarkOp";
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
      const std::string& pattern_str,
      DataType itype,
      const std::set<NodeDef*>& cluster,
      NodeDef* node);
};

} //namespace XDL_CORE_GRAPPLER_MARK_OP_FUSION_WORKER

#endif


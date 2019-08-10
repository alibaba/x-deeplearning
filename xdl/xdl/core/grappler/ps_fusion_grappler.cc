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

#include <memory>

#include "xdl/core/framework/grappler.h"
#include "xdl/core/grappler/hash_pull_fusion_worker.h"
#include "xdl/core/grappler/hash_push_fusion_worker.h"
#include "xdl/core/grappler/hash_statis_fusion_worker.h"
#include "xdl/core/grappler/mark_op_fusion_worker.h"

namespace xdl {

class PsFusionGrappler : public Grappler {
 public:
  Status Process(GraphDef* graph, OutputSpec* output) override {
    std::vector<std::unique_ptr<FusionWorker> > workers;
    workers.emplace_back(new HashStatisFusionWorker("pv"));
    workers.emplace_back(new HashStatisFusionWorker("click"));
    workers.emplace_back(new HashPullFusionWorker);
    workers.emplace_back(new HashPushFusionWorker);
    workers.emplace_back(new MarkOpFusionWorker);
    for (auto& worker: workers) {
      XDL_CHECK_STATUS(worker->Process(graph, output));
    }

    return Status::Ok();
  }
};

}  // namespace xdl

XDL_REGISTER_GRAPPLER(5000, xdl::PsFusionGrappler);


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

#include "xdl/core/framework/executor.h"

#include <vector>

#include "xdl/core/framework/graph_cache.h"
#include "xdl/core/framework/grappler.h"
#include "xdl/core/framework/graph_builder.h"

namespace xdl {

void Executor::Run(const GraphDef& graph,
                   const OutputSpec& output,
                   const RunOption& run_option,
                   Callback done) {
  auto graph_creator = 
  [](const GraphDef& def, const OutputSpec& output, Graph** g) -> Status {
    GraphDef real_def = def;
    OutputSpec real_output = output;
    XDL_CHECK_STATUS(
        GrapplerRegistry::Get()->Process(&real_def, &real_output));
    std::unique_ptr<Graph> ret(new Graph);
    GraphBuilder builder(real_def, real_output, ret.get());
    XDL_CHECK_STATUS(builder.Build());
    *g = ret.release();
    return Status::Ok();
  };
  Graph* g;
  Status build_status =
    GraphCache::Get()->GetGraph(graph_creator, graph, output, &g);
  if (!build_status.IsOk()) {
    done(build_status, std::vector<Tensor>(), SimpleExecutor::ExtraInfo());
    return;
  }
  SimpleExecutor::Run(g, run_option, done, thread_pool_);
}

}  // namespace xdl


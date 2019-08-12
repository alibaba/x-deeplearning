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

#ifndef XDL_CORE_FRAMEWORK_GRAPPLER_H_
#define XDL_CORE_FRAMEWORK_GRAPPLER_H_

#include <map>
#include <functional>

#include "xdl/core/lib/status.h"
#include "xdl/core/lib/singleton.h"
#include "xdl/core/framework/graph_def.h"

namespace xdl {

class Grappler {
 public:
  virtual ~Grappler() {}
  virtual Status Process(GraphDef* graph, OutputSpec* output) = 0;
};

class GrapplerRegistry : public Singleton<GrapplerRegistry> {
 public:
  void RegisterGrappler(int priority, Grappler* grappler) {
    grapplers_.insert({priority, grappler});
  }
  Status Process(GraphDef* graph, OutputSpec* output) {
    for (auto&& item : grapplers_) {
      XDL_CHECK_STATUS(item.second->Process(graph, output));
    }
    return Status::Ok();
  }
 private:
  std::multimap<int, Grappler*, std::greater<int>> grapplers_;
};

class GrapplerRegistryHelper {
 public:
  GrapplerRegistryHelper(int priority, Grappler* grappler) {
    GrapplerRegistry::Get()->RegisterGrappler(priority, grappler);
  }
};

}  // namespace xdl

#define XDL_REGISTER_GRAPPLER(priority, type)                     \
  XDL_REGISTER_GRAPPLER_UNIQ_HELPER(__COUNTER__, priority, type)
#define XDL_REGISTER_GRAPPLER_UNIQ_HELPER(ctr, priority, type)    \
  XDL_REGISTER_GRAPPLER_UNIQ(ctr, priority, type)
#define XDL_REGISTER_GRAPPLER_UNIQ(ctr, priority, type)           \
  static ::xdl::GrapplerRegistryHelper __register_grappler__##ctr \
  __attribute__((unused)) = ::xdl::GrapplerRegistryHelper(priority, new type)

#endif  // XDL_CORE_FRAMEWORK_GRAPPLER_H_


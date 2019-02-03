/*!
 * \file optimizer.h
 * \brief The optimizer.
 */
#pragma once

#include <memory>
#include <mutex>

#include "blaze/common/common_defines.h"
#include "blaze/optimizer/pass.h"

namespace blaze {

class Optimizer {
 public:
  static Optimizer* Get();

  Optimizer() { }

  NetDef RunPass(const NetDef& net_def) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Pass>>& all_pass = PassRegisterer::Get()->pass;
    size_t pre_op_num = net_def.op_size();
    NetDef nd = net_def;
    for (auto& pass : all_pass) {
      switch (pass->pass_type()) {
        case kGraph:
          nd = pass->RunPass(nd);
          break;
      }
    }
    size_t later_op_num = nd.op_size();
    LOG_INFO("optimizer op num: %u -> %u", pre_op_num, later_op_num);
    return nd;
  }

  NetDef RunPass(const NetDef& net_def, const char* pass_name) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Pass>>& all_pass = PassRegisterer::Get()->pass;
    size_t pre_op_num = net_def.op_size();
    NetDef nd = net_def;
    for (auto& pass : all_pass) {
      switch (pass->pass_type()) {
        case kGraph:
          if (pass->name() == pass_name) {
            nd = pass->RunPass(nd);
          }
          break;
      }
    }
    return nd;
  }

  NetDef RunPass(const NetDef& net_def, Workspace* ws) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Pass>>& all_pass = PassRegisterer::Get()->pass;
    NetDef nd = net_def;
    for (auto& pass : all_pass) {
      switch (pass->pass_type()) {
        case kWorkspace:
          nd = pass->RunPass(nd, ws);
          break;
      }
    }
    return nd;
  }

 protected:
  std::mutex mutex_;
  DISABLE_COPY_AND_ASSIGN(Optimizer);
};

}  // namespace blaze


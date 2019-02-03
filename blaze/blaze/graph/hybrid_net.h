/*
 * \file hybrid_net.h 
 * \brief The hybrid net for cross device execution. 
 */
#ifndef BLAZE_GRAPH_HYBRID_NET_H_
#define BLAZE_GRAPH_HYBRID_NET_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "blaze/graph/net.h"
#include "blaze/scheduler/scheduler_manager.h"
#include "blaze/scheduler/structured_batching.h"

namespace blaze {

class HybridNet : public Net {
 public:
  HybridNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);

  std::vector<OperatorBase*> GetOperators() override {
    std::vector<OperatorBase*> op_list;
    return op_list;
  }

  virtual bool Run() override;
  // async interface, just put the cb, first sub net and process func into scheduler 
  virtual bool Run(const PredictorCallback&& cb) override;
  virtual std::vector<std::string> GetTopoBlobName() const override;

 protected:
  DISABLE_COPY_AND_ASSIGN(HybridNet);

 private:
  // recursive func to run sub net 
  bool DoRun(Net* net, const PredictorCallback&& cb); 
  void MergeInputBlobs();
  void MergeOutputBlobs();
  void MergeTotalBlobs();

  std::vector<std::unique_ptr<Net>> sub_nets_;
  // <cur Net, next Net>
  std::unordered_map<Net*, Net*> topo_next_net_;
  std::shared_ptr<SchedulerManager<AsyncTask>> scheduler_manager_; 

  // about batching 
  StructuredBatching batching_;
};

} // namespace blaze

#endif  // BLAZE_GRAPH_HYBRID_NET_H_

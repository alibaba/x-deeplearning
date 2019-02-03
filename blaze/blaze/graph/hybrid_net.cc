/*
 * \file hybrid_net.cc 
 * \brief The hybrid net for cross device execution. 
 */

#include "blaze/graph/hybrid_net.h"
#include "blaze/graph/transform/cross_device_graph_manager.h"
#include "blaze/graph/simple_net.h"

using std::vector;

namespace blaze {

HybridNet::HybridNet(const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws) : Net(net_def, ws) {
  // build all sub nets
  CrossDeviceGraphManager manager(*net_def);  
  // rewrite graph and split graph
  manager.Transform();
  const std::vector<NetDef>& net_defs = manager.GetNetDefs();
  LOG_DEBUG("HybridNet, net def size:%u", net_defs.size());
  //TODO
  //NetDefHelper::SaveNetDefToTextFile("sub_graph", &(net_defs[0]));
  for (auto& sub_net_def : net_defs) {
    // iterate all sub net_def to build sub net
    // use SimpleNet by default   
    auto cur_net_def = std::make_shared<const NetDef>(sub_net_def);
    sub_nets_.emplace_back(new SimpleNet(std::move(cur_net_def), ws));     
  }
  
  // build sub net dependency
  if (sub_nets_.size() > 1) {
    for (size_t i = 0; i < sub_nets_.size() - 1; ++i) {
      topo_next_net_.emplace(sub_nets_[i].get(), sub_nets_[i + 1].get());
    }
  }

  // merge external_input_blob and external_output_blob
  MergeInputBlobs();
  MergeOutputBlobs();
  MergeTotalBlobs();

  // NOTE: SchedulerManager.Init should be invoked at first
  // when current process starts
  scheduler_manager_ = SchedulerManager<AsyncTask>::Instance();

  // construct hybrid net's input and output blob map
  for (auto& net : sub_nets_) {
    for (auto& op : net->operators()) {
      set_input_blob(op->operator_def(), op);
      set_output_blob(op->operator_def(), op);
    }
  }
}

std::vector<std::string> HybridNet::GetTopoBlobName() const {
  std::vector<std::string> names;
  std::set<std::string> sets;
  for (auto& net : sub_nets_) {
    auto new_names = net->GetTopoBlobName();
    for (const auto& new_name : new_names) {
      if (sets.count(new_name) == 0) {
        sets.insert(new_name);
        names.push_back(new_name);
      }
    }
  }
  return names;
}

void HybridNet::MergeInputBlobs() {
  for (auto& input : external_input_) {
    for (size_t i = 0; i < sub_nets_.size(); ++i) {
      auto blob = sub_nets_[i]->external_input_blob(input);
      if (blob) {
        external_input_blob_.emplace(input, blob);
        break;
      }
    }  
  }
}

void HybridNet::MergeOutputBlobs() {
  for (auto& output : external_output_) {
    for (size_t i = 0; i < sub_nets_.size(); ++i) {
      auto blob = sub_nets_[i]->external_output_blob(output);
      if (blob) {
        external_output_blob_.emplace(output, blob);
        break;
      }
    }  
  }
}

void HybridNet::MergeTotalBlobs() {
  for (size_t i = 0; i < sub_nets_.size(); ++i) {
    auto& blob_map = sub_nets_[i]->net_blob_map();
    net_blob_map_.insert(blob_map.begin(), blob_map.end());
  }
}

bool HybridNet::DoRun(Net* net, const PredictorCallback&& cb) {
  if (nullptr == net || nullptr == cb) {
    return false; 
  }
  
  // build AsyncTask
  std::unique_ptr<AsyncTask> async_task(new AsyncTask(net, this, std::move(cb)));
  auto& device_option = net->device_option();
  LOG_DEBUG("Net device type:%d device id:%d is pipe:%d",
      device_option.device_type(), device_option.device_id(), device_option.is_pipe());
  Scheduler* scheduler = scheduler_manager_->GetScheduler(device_option);
  if (nullptr == scheduler) {
    LOG_ERROR("Cannot get scheduler, maybe SchedulerManager.Init has not been invoked"); 
    return false;
  }

  // check the actual type of scheduler 
  SimpleScheduler<AsyncTask>* simple_scheduler = dynamic_cast<SimpleScheduler<AsyncTask>*>(scheduler);
  if (nullptr == simple_scheduler) {
    // it's BatchScheduler, or else return false 
    batching::SharedBatchScheduler<AsyncTask>* batch_scheduler =
      dynamic_cast<batching::SharedBatchScheduler<AsyncTask>*>(scheduler);   
    if (nullptr == batch_scheduler) {
      LOG_ERROR("Unsupported scheduler type");
      return false;
    }
    // invoke Schedule of BatchScheduler
    auto queue = scheduler_manager_->GetBatchedQueue(&(net_def()),
        [] (std::unique_ptr<batching::Batch<AsyncTask>> batch_tasks) {
            if (0 == batch_tasks->num_tasks()) {
              LOG_ERROR("None task to process");
              return;
            }
            LOG_DEBUG("Batch tasks num:%d", batch_tasks->num_tasks());
            AsyncTask* first_task = batch_tasks->mutable_task(0);
            HybridNet* first_task_parent_net = dynamic_cast<HybridNet*>(first_task->parent_net);

            vector<Net*> nets;
            for (int i = 0; i < batch_tasks->num_tasks(); ++i) {
              AsyncTask* task = batch_tasks->mutable_task(i);
              nets.push_back(task->net);
            }

            Net* merged_net;
            bool is_success = first_task_parent_net->batching_.Merge(nets,
                &merged_net);
            if (!is_success) {
              LOG_ERROR("Merge nets failed");
              return;
            }
            
            // only running the merged net 
            merged_net->Run();

            // split nets
            is_success = first_task_parent_net->batching_.Split(
                merged_net, &nets);
            if (!is_success) {
              LOG_ERROR("Split merged net to multi nets failed");
              return;
            }
            // invoke callback
            for (int i = 0; i < batch_tasks->num_tasks(); ++i) {
              AsyncTask* task = batch_tasks->mutable_task(i);
              (task->cb)(); 
            }
          }, batch_scheduler);
    if (nullptr == queue) {
      LOG_ERROR("GetBatchedQueue failed");
      return false;
    }
    if (batching::Status::kOk != queue->Schedule(&async_task)) {
      LOG_ERROR("Shared batching schedule failed");
      return false;
    }
  } else {
    // it's SimpleScheduler
    // declare a process func 
    static auto process_func = std::make_shared<SimpleScheduler<AsyncTask>::ProcessFunc>
      ([] (std::unique_ptr<AsyncTask> task) {
        HybridNet* task_parent_net = dynamic_cast<HybridNet*>(task->parent_net);
        // invoke synchronous interface net.Run 
        task->net->Run();

        // invoke next scheduler.Schedule if such scheduler exists
        auto it = task_parent_net->topo_next_net_.find(task->net);
        if (it != task_parent_net->topo_next_net_.end()) {
          task_parent_net->DoRun(it->second, std::move(task->cb));
        } else {
          // simply invoke callback
          (task->cb)();            
        }  
      });  
    simple_scheduler->Schedule(std::move(async_task), process_func); 
  }
  return true;
}

bool HybridNet::Run(const PredictorCallback&& cb) {
  // always use the first sub net as starting point
  // because sub_nets_ has been topological sorted
  return DoRun(sub_nets_[0].get(), std::move(cb)); 
}

bool HybridNet::Run() {
  Semaphore semaphore;
  if (!Run([&semaphore] { semaphore.notify(); })) {
    return false;
  }
  semaphore.wait();
  return true;
}

REGISTER_NET(hybrid, HybridNet);

} // namespace blaze

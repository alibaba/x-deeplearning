/*
 * \file workspace.h
 * \desc The run-time workspace
 */
#pragma once

#include <mutex>
#include <set>

#include "blaze/common/context.h"
#include "blaze/common/blob.h"
#include "blaze/common/log.h"
#include "blaze/proto/blaze.pb.h"
#include "blaze/store/sparse_puller.h"
#include "blaze/common/func.h"
#include "blaze/scheduler/scheduler_manager.h"

namespace blaze {

class Net;
using store::SparsePuller;
using store::SparsePullerCreationRegisterer;

// A model has a Workspace instance. The weights are shared by all Predict
// threads. And each threads has its's own input and output Blob.
class Workspace {
 public:
  virtual ~Workspace() {
    Clear();
  }

  // Create a constant fill blob, which is usually the model parameter blob.
  Blob* CreateConstantBlob(const std::string& name, const DeviceOption& device_option,
                           bool* newblob = nullptr) {
    if (!HasConstantBlob(name)) {
      constant_fill_blob_map_[name] = new Blob(device_option);
      if (newblob) *newblob = true;
    } else {
      if (newblob) *newblob = false;
    }
    return reinterpret_cast<Blob*>(GetConstantBlob(name));
  }

  // Create or get the blob of name.
  Blob* CreateBlob(const std::string& name, const DeviceOption& device_option,
                   bool* newblob = nullptr) {
    if (HasConstantBlob(name)) {
      if (newblob) *newblob = false;
      return GetConstantBlob(name);
    }
    if (!HasBlob(name)) {
      blob_map_[name] = new Blob(device_option);
      if (newblob) *newblob = true;
    } else {
      if (newblob) *newblob = false;
    }
    return reinterpret_cast<Blob*>(GetBlob(name));
  }
  void SetBlob(const std::string& name, Blob* blob) {
    if (!HasBlob(name)) {
      blob_map_[name] = blob;
    }
  }
  
  // Set the sparse puller
  void SetSparsePuller(std::shared_ptr<SparsePuller> sparse_puller) {
    sparse_puller_ = sparse_puller;
  }
  // Get the sparse puller
  std::shared_ptr<SparsePuller>& GetSparsePuller() {
    return sparse_puller_;
  }

  // Initialize the NetDef.
  // init the input and output data type.
  void Init(const NetDef& net_def);

  // Creates a network with the given NetDef, and returns the pointer to the
  // network.
  std::shared_ptr<Net> CreateNet();

  // Only used for small model creation(Mainly Text Pb DNN Model), such as: Testing model
  // NOTE: not used in production env.
  std::shared_ptr<Net> CreateNet(const char* net_file);

  // Return the input data type of input_name
  DataType input_data_type(const std::string& input_name) const {
    const auto& iter = input_data_type_map_.find(input_name);
    CHECK(iter != input_data_type_map_.end(), "input_name: %s data_type not defined",
          input_name.c_str());
    return iter->second;
  }
  // Return the output data type of output_name.
  DataType output_data_type(const std::string& output_name) const {
    const auto& iter = output_data_type_map_.find(output_name);
    CHECK(iter != output_data_type_map_.end(), "output_name: %s data_type not defined",
          output_name.c_str());
    return iter->second;
  }

  std::shared_ptr<NetDef>& net_def() { return net_def_; }

  // Return the input info.
  const ValueInfo& input_info(const std::string& input_name) const {
    const auto& iter = input_info_map_.find(input_name);
    CHECK_TRUE(iter != input_info_map_.end(), "input_name=", input_name, " is not input name");
    return iter->second;
  }

  bool has_input_info(const std::string& input_name) const {
    const auto& iter = input_info_map_.find(input_name);
    return iter != input_info_map_.end();
  }

 protected:
  // Recycle blob map for the next net handle creation. each net handle has its
  // own input/output context.
  void RecycleBlobMap();

  Blob* GetBlob(const std::string& name) {
    auto iter = blob_map_.find(name);
    CHECK(iter != blob_map_.end(), "blob %s not existed", name.c_str());
    return iter->second;
  }
  bool HasBlob(const std::string& name) {
    auto iter = blob_map_.find(name);
    if (iter == blob_map_.end()) {
      return false;
    }
    return true;
  }

  Blob* GetConstantBlob(const std::string& name) {
    auto iter = constant_fill_blob_map_.find(name);
    CHECK(iter != constant_fill_blob_map_.end(), "blob %s not existed", name.c_str());
    return iter->second;
  }
  bool HasConstantBlob(const std::string& name) {
    auto iter = constant_fill_blob_map_.find(name);
    if (iter == constant_fill_blob_map_.end()) {
      return false;
    }
    return true;
  }
  void InitInputOutputDataTypeMap();
  void InitInputInfoMap();

  void Clear() {
    // Clear memory
    std::set<Blob*> removed;
    for (auto& item : blob_map_) {
      if (removed.count(item.second) == 0) {
        delete item.second;
        removed.insert(item.second);
      }
    }
    blob_map_.clear();
    // delete the graphs's total fill blob
    for (auto& item : constant_fill_blob_map_) {
      delete item.second;
    }
    constant_fill_blob_map_.clear();
    // delete the previous created blobs
    for (auto& item : blob_recycle_bins_) {
      delete item;
    }
    blob_recycle_bins_.clear();

    // clear input output map
    input_data_type_map_.clear();
    output_data_type_map_.clear();
  
    // release batched queue from SchedulerManager
    scheduler_manager_->ReleaseBatchedQueue(net_def_.get()); 
  }

  // The constant fill blob map is shared by all predict threads.
  std::map<std::string, Blob*> constant_fill_blob_map_;
  // Each threads has it's own blob map.
  std::map<std::string, Blob*> blob_map_;

  // The blob recycle bin
  std::set<Blob*> blob_recycle_bins_;

  std::shared_ptr<NetDef> net_def_;
  std::map<std::string, DataType> input_data_type_map_;
  std::map<std::string, DataType> output_data_type_map_;
  std::unordered_map<std::string, ValueInfo> input_info_map_;

  // sparse puller
  std::shared_ptr<SparsePuller> sparse_puller_;

  std::mutex mutex_;

  std::shared_ptr<SchedulerManager<AsyncTask>> scheduler_manager_;
};

}  // namespace blaze


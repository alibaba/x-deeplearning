/*
 * \file batching.cc 
 * \brief high-level batching func to batch multi nets 
 */

#include "blaze/scheduler/batching.h"
#include "blaze/common/log.h"
#include "blaze/common/context.h"

namespace blaze {

Batching::Batching() {
}

Batching::~Batching() {

}

void Batching::BackupFirstDim(const Net* net) {
  blob_first_dim_.clear();
  for (auto blob_pair : net->external_input_blob()) {
    if (blob_pair.second->device_type() == kCUDA) {
      blob_first_dim_[blob_pair.first] = blob_pair.second->dim(0);
    }
  }
}

void Batching::RestoreFirstDim(Net* net) {
  for (auto& blob_pair : net->external_input_blob()) {
    if (blob_pair.second->device_type() == kCUDA) {
      auto dims = blob_pair.second->shape();
      dims[0] = blob_first_dim_[blob_pair.first];
      blob_pair.second->Reshape(dims);
    }
  }
}

void Batching::GetNonIndicatorBlobs(const Net* net,
    std::unordered_map<std::string, Blob*>* blobs_map) {
  for (auto input_name : net->external_input()) {
    if (net->has_external_input_info(input_name)) {
      auto& value_info = net->external_input_info(input_name); 
      if (value_info.has_input_type() &&
        value_info.input_type() == kInputIndicator) {
        continue;
      }   
    }
    
    auto blob = net->external_input_blob(input_name);
    if (blob && blob->device_type() == kCUDA) {
      blobs_map->emplace(input_name, blob);
    } else {
      LOG_INFO("Blob name:%s belongs to cpu device", input_name.c_str());
    }
  } 
}

TIndex Batching::CalcSumOfFirstDim(std::vector<Net*>& nets,
    const std::string& blob_name) {
  TIndex first_dim_sum = 0; 
  for (auto net : nets) {
    auto blob = net->external_input_blob(blob_name);
    first_dim_sum += blob->dim(0);
  } 
  return first_dim_sum;
}

void Batching::BackupData(const Net* net, const std::string& blob_name,
    Blob* backup_blob) {
#if USE_CUDA
  auto blob = net->external_input_blob(blob_name);
  TIndex data_size = blob->size() * DataTypeSize(blob->data_type());
  backup_blob->Reshape(blob->shape());
  CUDAContext context(net->device_option()); 
  cudaStream_t stream = context.cuda_stream();
  CUDA_CHECK(cudaMemcpyAsync(backup_blob->data(), blob->data(),
      data_size, cudaMemcpyDeviceToDevice, stream));
  context.FinishDeviceComputation();
#endif // USE_CUDA
}

void Batching::ReshapeMergedNet(std::vector<Net*>& nets,
    const std::string& blob_name, Net* merged_net) {
  TIndex first_dim_sum = CalcSumOfFirstDim(nets, blob_name); 
  auto blob = merged_net->external_input_blob(blob_name);
  auto merged_dims = blob->shape();
  merged_dims[0] = first_dim_sum;
  blob->Reshape(merged_dims);
} 

void Batching::RestoreData(const Blob& backup_blob, Net* net,
    const std::string& blob_name, TIndex* offset) {
#if USE_CUDA
  auto blob = net->external_input_blob(blob_name);
  CUDAContext context(net->device_option()); 
  cudaStream_t stream = context.cuda_stream();
  TIndex data_size = backup_blob.size() * DataTypeSize(backup_blob.data_type());
  CUDA_CHECK(cudaMemcpyAsync(blob->data(), backup_blob.data(),
      data_size, cudaMemcpyDeviceToDevice, stream));
  *offset = data_size;
  context.FinishDeviceComputation();
#endif // USE_CUDA
}

void Batching::CopyDataToMergedNet(const Net* net, const std::string& blob_name,
      CUDAContext& context, Net* merged_net, TIndex* offset) {
#if USE_CUDA
  auto src_blob = net->external_input_blob(blob_name);
  cudaStream_t stream = context.cuda_stream();
  auto dst_blob = merged_net->external_input_blob(blob_name); 
  TIndex data_size = src_blob->size() * DataTypeSize(src_blob->data_type());
  CUDA_CHECK(cudaMemcpyAsync(dst_blob->as<char>() + *offset, src_blob->data(),
      data_size, cudaMemcpyDeviceToDevice, stream));
  *offset += data_size;
#endif // USE_CUDA
}

bool Batching::Merge(std::vector<Net*>& src_nets, Net** dst_net) {
#if USE_CUDA
  // always choose first net as dst_net  
  if (0 == src_nets.size() && nullptr == dst_net) {
    LOG_ERROR("Input src nets is empty, or dst_net is nullptr");
    return false;
  }
  *dst_net = src_nets[0];  

  BackupFirstDim(*dst_net); 

  std::unordered_map<std::string, Blob*> non_indicator_blobs_map;
  GetNonIndicatorBlobs(*dst_net, &non_indicator_blobs_map); 
  CUDAContext context((*dst_net)->device_option()); 
  Blob backup_blob((*dst_net)->device_option());
  for (auto& blob_pair : non_indicator_blobs_map) {
    // backup old data of dst net
    BackupData(*dst_net, blob_pair.first, &backup_blob);
    // reshape dst net 
    ReshapeMergedNet(src_nets, blob_pair.first, *dst_net);
    // restore data of dst net
    TIndex dst_data_offset;
    RestoreData(backup_blob, *dst_net,
        blob_pair.first, &dst_data_offset);  
    // copy data from nets except the first net to dst net
    for (int i = 1; i < src_nets.size(); ++i) {
      CopyDataToMergedNet(src_nets[i], blob_pair.first, context,
          *dst_net, &dst_data_offset); 
    }
  }
  context.FinishDeviceComputation();

  // specially handle some inputs, such as: indicators
  HandleIndicators(src_nets);
#endif // USE_CUDA
  return true;
}

bool Batching::Split(Net* src_net, std::vector<Net*>* dst_nets) {
  if (nullptr == src_net || nullptr == dst_nets ||
      dst_nets->size() == 0) {
    LOG_ERROR("Wrong input params of DoSplit func");
    return false;
  }
  // for input blobs: restore first dim of src_net 
  RestoreFirstDim(src_net);
 
  if (level_lens_.size() == 0) {
    // NOTE: empty level_lens_ means that there is no need to split
    return true;
  }
  // for output blobs: always split src net by the lens of level 0 
  auto& lens = level_lens_[0];
  auto& output_blobs_map = src_net->external_output_blob();
  for (auto& blob_pair : output_blobs_map) {
    // reshape the external output blob of first net
    auto split_dims = blob_pair.second->shape();
    split_dims[0] = lens[0];
    // NOTE: shrinking size would not lead to release the original space
    blob_pair.second->Reshape(split_dims);  

    TIndex offset = blob_pair.second->size();
    // refreshape external output blob of nets except the first nets
    for (int i = 1; i < dst_nets->size(); ++i) {
      split_dims[0] = lens[i];
      auto blob = (*dst_nets)[i]->external_output_blob(blob_pair.first);
      blob->RefReshape(split_dims, blob_pair.second->as<char>()
          + offset * DataTypeSize(blob_pair.second->data_type()));    
      offset += blob->size();
    } 
  }

  return true;
}

} // namespace blaze

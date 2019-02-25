/*
 * \file structured_batching.cc 
 * \brief for batching structured data 
 */

#include <unordered_map>
#include <string>
#include <vector>

#include "blaze/scheduler/structured_batching.h"
#include "blaze/common/log.h"
#include "blaze/common/common_defines.h"

using std::unordered_map;
using std::string;
using std::vector;

namespace blaze {

StructuredBatching::StructuredBatching() {
}

void StructuredBatching::CopyDeviceToHost(std::vector<Net*>& nets,
    const std::string& blob_name, std::vector<TIndex>* lens,
    Blob* host_indicators) {
#if USE_CUDA
  TIndex total_size = 0; 
  for (auto& net : nets) {
    auto blob = net->external_input_blob(blob_name);
    total_size += blob->size();
  }
  host_indicators->Reshape({total_size});
  CUDAContext context(nets[0]->device_option()); 
  cudaStream_t stream = context.cuda_stream();
  TIndex offset = 0;
  for (auto& net : nets) {
    auto blob = net->external_input_blob(blob_name);
    TIndex cur_size = blob->size() * DataTypeSize(blob->data_type());
    CUDA_CHECK(cudaMemcpyAsync(host_indicators->as<char>() + offset,
          blob->data(), cur_size, cudaMemcpyDeviceToHost, stream));
    offset += cur_size; 
    lens->push_back(blob->size());
  }
  context.FinishDeviceComputation();
#endif // USE_CUDA
} 

void StructuredBatching::AdjustVals(const std::vector<TIndex>& lens,
    Blob* host_indicators) {
  int* indicators = host_indicators->as<int>();
  TIndex bias = 0;
  for (auto len : lens) {
    for (TIndex i = 0; i < len; ++i) {
      indicators[i] += bias;
    } 
    bias = indicators[len - 1] + 1; 
    indicators += len;
  } 
} 

void StructuredBatching::UpdateLevelLens(const std::vector<TIndex>& lens, int level_id) {
  for (auto len : lens) {
    level_lens_[level_id].push_back(len);
  }
} 

TIndex StructuredBatching::TotalLen(const std::vector<TIndex>& lens) {
  TIndex total_len = 0;
  for (auto len : lens) {
    total_len += len;
  }
  return total_len;
}

void StructuredBatching::ReshapeIndicator(const std::string& blob_name,
    TIndex total_len, Net* net) {
  auto blob = net->external_input_blob(blob_name);
  blob->Reshape({total_len});  
} 

void StructuredBatching::CopyHostToDevice(const std::string& blob_name,
    TIndex total_len, const Blob& host_indicators, Net* net) {
#if USE_CUDA
  auto blob = net->external_input_blob(blob_name);
  CUDAContext context(net->device_option()); 
  cudaStream_t stream = context.cuda_stream();
  CUDA_CHECK(cudaMemcpyAsync(blob->as<char>(), host_indicators.data(),
      total_len * DataTypeSize(blob->data_type()), cudaMemcpyHostToDevice, stream));
  context.FinishDeviceComputation();
#endif // USE_CUDA
}

bool StructuredBatching::HandleIndicators(std::vector<Net*>& nets) {
  if (0 == nets.size()) {
    LOG_ERROR("Input nets's size should be larger than 0");
    return false; 
  }
 
  // Check the input type of external input to find out indicators 
  unordered_map<int, vector<string>> level_indicators_map; 
  auto& net_def = nets[0]->net_def();
  for (int i = 0; i < net_def.external_input_size(); ++i) {
    if (net_def.external_input(i).has_input_type() &&
        net_def.external_input(i).input_type() == kInputIndicator) {
      level_indicators_map[net_def.external_input(i).level()].emplace_back(
          net_def.external_input(i).name());
    } 
  }
  // NOTE: flattened_batching is the special case of structured_batching
  // if there is no indicator, just regard it as flattened_batching
  if (0 == level_indicators_map.size()) {
    auto& blob_map = nets[0]->external_input_blob();
    // there is only one layer, which means that all the blobs' first 
    // dimension are the same in one net, so just regard the first blob's name
    // as representative
    if (blob_map.size() == 0) {
      // NOTE: if there is no external input blob, just do nothing
      return true; 
    }
    string one_name = blob_map.begin()->first;
    level_lens_.resize(1u);
    // update lens of each net      
    for (auto& net : nets) {
      auto input_blob = net->external_input_blob(one_name); 
      level_lens_[0].push_back(input_blob->dim(0)); 
    }
  } else {
    level_lens_.resize(level_indicators_map.size());
    // handle indicators level by level 
    DeviceOption device_option;
    device_option.set_device_type(kCPU);
    device_option.set_device_id(0);
    Blob host_indicators(device_option);
    for (auto& level_indicators : level_indicators_map) {
      // copy data from GPU to cpu
      std::vector<TIndex> lens;
      CopyDeviceToHost(nets, level_indicators.second.at(0),
          &lens, &host_indicators); 
      // adjust vals of indicators
      AdjustVals(lens, &host_indicators);
      UpdateLevelLens(lens, level_indicators.first);
      TIndex total_len = TotalLen(lens);
      for (auto& indicator_name : level_indicators.second) {
        // reshape blob in GPU
        ReshapeIndicator(indicator_name, total_len, nets[0]);    
        // copy vals from cpu to GPU
        CopyHostToDevice(indicator_name, total_len, host_indicators, nets[0]);
      }
    }
  }
  
  return true;
}

} // namespace blaze

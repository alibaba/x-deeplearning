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

#include "ps-plus/server/udf_manager.h"

namespace ps {
namespace server {

UdfChain::~UdfChain() {
  for (auto udf : udfs_) {
    delete udf;
  }
}

Status UdfChain::BuildFromDef(const UdfChainRegister& def) {
  // Calculate Input Size
  std::size_t input_size = 0;
  for (std::size_t i = 1; i < def.udfs.size(); i++) {
    for (auto input : def.udfs[i].inputs) {
      if (input.first == 0) {
        input_size = std::max(input_size, (std::size_t)input.second + 1);
      }
    }
  }
  input_size_ = input_size;

  // Calculate Every Node Input
  std::vector<std::vector<size_t>> output_nodes;
  size_t output_counter = 0;
  output_nodes.emplace_back();
  for (size_t i = 0; i < input_size; i++) {
    output_nodes[0].push_back(output_counter++);
  }
  for (std::size_t i = 1; i < def.udfs.size(); i++) {
    std::vector<size_t> indexed_input;
    for (auto input : def.udfs[i].inputs) {
      if (input.first < 0 || input.first >= (int)output_nodes.size() ||
          input.second < -1 || input.second >= (int)output_nodes[input.first].size()) {
        return Status::IndexOverflow("UdfChain BuildFromDef Index Overflow");
      }
      if (input.second == -1) {
        for (size_t id : output_nodes[input.first]) {
          indexed_input.push_back(id);
        }
      } else {
        indexed_input.push_back(output_nodes[input.first][input.second]);
      }
    }
    UdfRegistry* udf_reg = UdfRegistry::Get(def.udfs[i].udf_name);
    if (udf_reg == nullptr) {
      return Status::NotFound("UdfChain BuildFromDef udf not found: " + def.udfs[i].udf_name);
    }
    if (indexed_input.size() < udf_reg->InputSize()) {
      return Status::DataLoss("UdfChain BuildFromDef input size error");
    }
    std::vector<size_t> indexed_output;
    for (size_t j = 0; j < udf_reg->OutputSize(); j++) {
      indexed_output.push_back(output_counter++);
    }
    output_nodes.push_back(indexed_output);
    Udf* udf = udf_reg->Build(indexed_input, indexed_output);
    udfs_.push_back(udf);
  }

  // Calculate Outputs
  for (auto output : def.outputs) {
    if (output.first < 0 || output.first >= (int)output_nodes.size() ||
        output.second < -1 || output.second >= (int)output_nodes[output.first].size()) {
      return Status::IndexOverflow("UdfChain BuildFromDef Output Index Overflow");
    }
    if (output.second == -1) {
      for (size_t id : output_nodes[output.first]) {
        output_ids_.push_back(id);
      }
    } else {
      output_ids_.push_back(output_nodes[output.first][output.second]);
    }
  }

  return Status::Ok();
}

Status UdfChain::Process(UdfContext* ctx) {
  if (ctx->DataSize() < input_size_) {
    return Status::DataLoss("UdfChain Process Input Loss");
  }
  for (size_t i = input_size_; i < ctx->DataSize(); i++) {
    PS_CHECK_STATUS(ctx->SetData(i, nullptr, false));
  }
  for (Udf* udf : udfs_) {
    PS_CHECK_STATUS(udf->Run(ctx));
  }
  PS_CHECK_STATUS(ctx->ProcessOutputs(output_ids_));
  return Status::Ok();
}

UdfChain* UdfChainManager::GetUdfChain(size_t hash) {
  QRWLocker lock(rd_lock_, QRWLocker::kSimpleRead);
  auto iter = chain_map_.find(hash);
  if (iter == chain_map_.end()) {
    return nullptr;
  } else {
    return iter->second.get();
  }
}

Status UdfChainManager::RegisterUdfChain(const UdfChainRegister& def) {
  QRWLocker lock(rd_lock_, QRWLocker::kWrite);
  auto iter = chain_map_.find(def.hash);
  if (iter != chain_map_.end()) {
    return Status::Ok();
  }
  std::unique_ptr<UdfChain> chain(new UdfChain);
  PS_CHECK_STATUS(chain->BuildFromDef(def));
  chain_map_[def.hash] = std::move(chain);
  return Status::Ok();
}

}
}


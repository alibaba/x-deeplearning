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

#ifndef PS_PLUS_CLIENT_UDF_H_
#define PS_PLUS_CLIENT_UDF_H_

#include <vector>
#include <memory>
#include <string>
#include <unordered_map> 

#include "ps-plus/message/udf_chain_register.h"

namespace ps {
namespace client {

class UdfChain;

class UdfData {
  friend class UdfChain;
 private:
  struct UdfNode {
    std::string udf_name_;
    std::vector<UdfData> inputs_;
    UdfNode(const std::string& udf_name, const std::vector<UdfData>& inputs)
      : udf_name_(udf_name), inputs_(inputs) {}
  };

 public:
  explicit UdfData(int input) : node_(nullptr), id_(input) {}
  UdfData(const std::shared_ptr<UdfNode>& node, int id) : node_(node), id_(id) {}
  template<class... T>
  UdfData(std::string udf_name, T... args)
    : node_(new UdfNode(udf_name, std::vector<UdfData>( {
      args...
  }))), id_(-1) {}

  template<class... T>
  UdfData(int id, std::string udf_name, T... args)
    : node_(new UdfNode(udf_name, std::vector<UdfData>( {
    args...
  }))), id_(id) {}

  UdfData(std::string udf_name, const std::vector<UdfData>& args)
    : node_(new UdfNode(udf_name, args)), id_(-1) {}

  UdfData(int id, std::string udf_name, const std::vector<UdfData>& args)
    : node_(new UdfNode(udf_name, args)), id_(id) {}

  UdfData operator()(int id) const {
    return UdfData(this->node_, id);
  }

  std::string Name() const {
    return node_->udf_name_;
  }

  const std::vector<UdfData>& Inputs() const {
    return node_->inputs_;
  }

  int Id() const {
    return id_;
  }

 private:
  std::shared_ptr<UdfNode> node_;
  int id_;
};

class UdfChain {
 public:
  UdfChain(const UdfData& data) : hash_(0) {
    datas_.push_back(data);
    CalculateHash();
  }

  UdfChain(const std::vector<UdfData>& datas) : datas_(datas), hash_(0) {
    CalculateHash();
  }

  unsigned long long hash() const {
    return hash_;
  }

  UdfChainRegister BuildChainRegister() const;

 private:

  // Using RK Hashing Algorithm to hash following sequence
  // {
  // node0_input_size + kHash0, node0_udf_name, node0_input0_node, node0_input0_index, node0_input1_node, node0_input1_index, ...,
  // node1_input_size + kHash0, ...,
  // ...,
  // output_size + kHash1, output0_node, output0_index, output1_node, output1_index...
  // }
  void CalculateHash();

  // return the hash of {nodex_inputx_node, nodex_inputx_index}
  unsigned long long CollectNode(const UdfData& data);

  std::vector<UdfData> datas_;
  std::vector<UdfData::UdfNode*> nodes_;
  std::unordered_map<UdfData::UdfNode*, int> node_ids_;
  unsigned long long hash_;
};

class UdfDef {
 public:
  explicit UdfDef(const std::string& udf_name)
        : udf_name_(udf_name) {}
  template<class... T>
  UdfData operator()(T... args) {
    return UdfData(udf_name_, args...);
  }
 private:
  std::string udf_name_;
};

}
}

#endif

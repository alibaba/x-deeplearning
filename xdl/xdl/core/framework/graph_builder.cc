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

#include "xdl/core/framework/graph_builder.h"

#include <vector>
#include <string>

#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/framework/device_registry.h"

namespace xdl {

Status GraphBuilder::Build() {
  XDL_CHECK_STATUS(Prepare());
  XDL_CHECK_STATUS(BuildNodes());
  XDL_CHECK_STATUS(AppendOutputs());
  XDL_CHECK_STATUS(AddDeviceConverter());
  XDL_CHECK_STATUS(CheckDAG());
  XDL_CHECK_STATUS(CheckOutputOverflow());
  return Status::Ok();
}

Status GraphBuilder::Prepare() {
  for (size_t i = 0; i < def_.node.size(); i++) {
    // 0 for Source, 1 for Sink
    XDL_CHECK_COND(node_id_.insert({def_.node[i].name, i + 2}).second,
                   Status::ArgumentError("Node Name Redfined "
                                         + def_.node[i].name));
  }
  return Status::Ok();
}

Status GraphBuilder::BuildNodes() {
  // 0 for Source
  graph_->nodes.emplace_back();
  // 1 for Sink
  graph_->nodes.emplace_back();
  XDL_CHECK_STATUS(BuildSink(&graph_->nodes.back()));

  for (auto&& item : def_.node) {
    graph_->nodes.emplace_back();
    XDL_CHECK_STATUS(BuildNode(item, &graph_->nodes.back()));
  }
  return Status::Ok();
}

Status GraphBuilder::AddDeviceConverter() {
  for (auto&& item : graph_->nodes) {
    for (auto&& output : item.outputs) {
      if (output.output_id == Node::kDependency) {
        continue;
      }
      Device* src_device = item.arg.device;
      Device* dst_device = graph_->nodes[output.node_id].arg.device;
      if (src_device != dst_device) {
        XDL_CHECK_STATUS(CreateDeviceConverter(src_device, dst_device));
      }
      for (int input_id = 0; input_id < graph_->nodes[output.node_id].inputs.size(); ++input_id) {
        if (input_id >= graph_->nodes[output.node_id].arg.input_devices.size()) break;
        dst_device = graph_->nodes[output.node_id].arg.input_devices[input_id];
        if (dst_device != nullptr && src_device != dst_device) {
          XDL_CHECK_STATUS(CreateDeviceConverter(src_device, dst_device));
        }
      }
    }
  }
  return Status::Ok();
}

Status GraphBuilder::BuildNode(const NodeDef& def, Node* node) {
  auto attr = def.attr;
  node->name = def.name;
  node->op_name = def.op;
  XDL_CHECK_STATUS(OpDefine::Get()->PrepareOpKernelContext(
        def.op, &attr, &node->arg));
  XDL_CHECK_STATUS(DeviceRegistry::Get()->GetDevice(
        def.device, &node->arg.device));
  for (auto&& item : def.input) {
    node->inputs.emplace_back();
    XDL_CHECK_STATUS(ParseInput(item, &node->inputs.back()));
  }
  node->input_size = node->arg.input_name.size();
  for (const std::string& input_dev_desc : def.input_dev_descs) {
    if (input_dev_desc == "CPU") {
      DeviceDef device;
      device.device_name = "CPU";
      node->arg.input_devices.emplace_back();
      XDL_CHECK_STATUS(DeviceRegistry::Get()->GetDevice(
            device, &node->arg.input_devices[node->arg.input_devices.size() - 1]));
    } else {
      node->arg.input_devices.emplace_back(nullptr);
    }
  }
  OpKernelBase* op;
  XDL_CHECK_STATUS(OpRegistry::Get()->CreateKernel(
        def, node->arg.device->DeviceType(), &op));
  node->op.reset(op);
  OpKernelConstruction construction(attr, node->arg.device);
  XDL_CHECK_STATUS(node->op->Init(&construction));
  return Status::Ok();
}

Status GraphBuilder::BuildSink(Node* node) {
  node->name = "_Sink";
  size_t input_size = 0;
  for (auto&& item : output_.output) {
    node->inputs.emplace_back();
    XDL_CHECK_STATUS(ParseInput(item, &node->inputs.back()));
    if (node->inputs.back().output_id != Node::kDependency) {
      input_size++;
    }
  }
  node->input_size = input_size;
  XDL_CHECK_STATUS(DeviceRegistry::Get()->GetDevice(
        output_.output_device, &node->arg.device));
  return Status::Ok();
}

Status GraphBuilder::AppendOutputs() {
  for (size_t i = 0; i < graph_->nodes.size(); i++) {
    auto& item = graph_->nodes[i];
    size_t input_size = 0;
    for (auto& input : item.inputs) {
      int input_id;
      if (input.output_id == Node::kDependency) {
        input_id = Node::kDependency;
      } else {
        input_id = input_size++;
      }
      graph_->nodes[input.node_id].outputs.push_back(Node::Output{
          .output_id = input.output_id,
          .node_id = static_cast<int>(i),
          .input_id = input_id});
    }
    XDL_CHECK_COND(item.input_size == input_size,
                   Status::ArgumentError("Node Input Size Check Error " + item.name
                                         + ", " + std::to_string(item.input_size) + " != " + std::to_string(input_size)));
  }
  XDL_CHECK_STATUS(BuildSource(&graph_->nodes[Graph::kSource]));
  return Status::Ok();
}

Status GraphBuilder::BuildSource(Node* node) {
  node->name = "_Source";
  for (int i = 0; i < static_cast<int>(graph_->nodes.size()); i++) {
    if (i == Graph::kSource || i == Graph::kSink) {
      continue;
    }
    if (graph_->nodes[i].inputs.empty()) {
      node->outputs.push_back(Node::Output{
          .output_id = Node::kDependency,
          .node_id = i,
          .input_id = Node::kDependency
      });
    }
  }
  return Status::Ok();
}

Status GraphBuilder::ParseInput(const std::string& spec, Node::Input* result) {
  if (!spec.empty() && spec[0] == '^') {
    std::string name = spec.substr(1);
    auto iter = node_id_.find(name);
    XDL_CHECK_COND(iter != node_id_.end(),
                   Status::ArgumentError("Node Input Error, "
                                         "input node not found " + spec));
    result->node_id = iter->second;
    result->output_id = Node::kDependency;
    return Status::Ok();
  } else {
    size_t pos = spec.find(':');
    XDL_CHECK_COND(pos != std::string::npos,
                   Status::ArgumentError("Node Input Error, "
                                         ": not found " + spec));
    std::string name = spec.substr(0, pos);
    std::string id_str = spec.substr(pos + 1);
    auto iter = node_id_.find(name);
    XDL_CHECK_COND(iter != node_id_.end(),
                   Status::ArgumentError("Node Input Error, "
                                         "input node not found " + spec));
    int id = std::atoi(id_str.c_str());
    XDL_CHECK_COND(std::to_string(id) == id_str,
                   Status::ArgumentError("Node Input Error, "
                                         "id is not a number " + spec));
    XDL_CHECK_COND(id >= 0,
                   Status::ArgumentError("Node Input Error, "
                                         "id must not be negative."));
    result->node_id = iter->second;
    result->output_id = id;
    return Status::Ok();
  }
}

Status GraphBuilder::CreateDeviceConverter(Device* src, Device* dst) {
  std::pair<Device*, Device*> key(src, dst);
  auto iter = graph_->device_converter.find(key);
  if (iter != graph_->device_converter.end()) {
    return Status::Ok();
  }
  DeviceConverter* converter = DeviceConverterRegistry::Instance()
      ->Get(src->DeviceType(), dst->DeviceType());
  XDL_CHECK_COND(converter != nullptr,
                 Status::Internal("Device Converter is not implemented."
                                  + src->DeviceType() + " -> "
                                  + dst->DeviceType()));
  graph_->device_converter[key] = converter;
  return Status::Ok();
}

Status GraphBuilder::CheckDAG() {
  std::vector<int> ref;
  std::vector<int> queue;
  for (size_t i = 0; i < graph_->nodes.size(); i++) {
    auto& item = graph_->nodes[i];
    ref.push_back(item.inputs.size());
    if (item.inputs.size() == 0) {
      queue.push_back(i);
    }
  }
  for (size_t i = 0; i < queue.size(); i++) {
    auto& item = graph_->nodes[queue[i]];
    for (auto& output : item.outputs) {
      if (--ref[output.node_id] == 0) {
        queue.push_back(output.node_id);
      }
    }
  }
  XDL_CHECK_COND(queue.size() == graph_->nodes.size(),
                 Status::ArgumentError("Graph Error, Not a DAG"));
  return Status::Ok();
}

Status GraphBuilder::CheckOutputOverflow() {
  for (auto&& node : graph_->nodes) {
    for (auto&& output : node.outputs) {
      XDL_CHECK_COND(
          output.output_id < static_cast<int>(node.arg.output_name.size()),
          Status::ArgumentError("Input Overflow for node "
                                + graph_->nodes[output.node_id].name + " "
                                + node.name + ":"
                                + std::to_string(output.output_id)));
    }
  }
  return Status::Ok();
}

}  // namespace xdl


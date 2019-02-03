/*
 * \file operator.cc
 * \desc The base operator.
 */
#include "blaze/operator/operator.h"

namespace blaze {

OperatorBase::OperatorBase(const OperatorDef& def, Workspace* workspace, bool alloc_input_output_blob) :
    def_(def),
    workspace_(workspace),
    device_option_(def.has_device_option() ? def.device_option() : DeviceOption()),
    event_(make_unique<Event>(device_option_)) {
  type_ = def_.type();
  name_ = def_.name();
  
  OpSchema* schema = OpSchemaRegistry::Schema(type_);
  CheckSchema(schema);

  if (alloc_input_output_blob) {
    AllocInputOutputBlob(def);
  }
}

void OperatorBase::AllocInputOutputBlob(const OperatorDef& def) {
  OpSchema* schema = OpSchemaRegistry::Schema(this->type_);

  // allocate op input
  bool newblob = false;
  for (const std::string& input_str : def.input()) {
    Blob* blob = workspace_->CreateBlob(input_str, device_option_, &newblob);
    CHECK(blob != nullptr, "op: %s has a non-existing input blob",
          def.type().c_str(), input_str.c_str());
    inputs_.push_back(blob);
    if (newblob) { // must be input
      blob->set_data_type(workspace_->input_data_type(input_str));
    }
  }
  LOG_DEBUG("name_=%s type_=%s inputs_.size()=%u",
            name_.c_str(), type_.c_str(), inputs_.size());

  // infer tensor type of output.
  std::vector<DataType> input_types;
  for (auto item : inputs_) {
    input_types.push_back(static_cast<DataType>(item->data_type()));
  }
  std::vector<DataType> output_types = schema->InferType(def_, input_types);

  for (size_t k = 0; k < def.output_size(); ++k) {
    const std::string& output_str = def.output(k);
    /*NOTE: Can not Inplace directly */
    Blob* blob = workspace_->CreateBlob(output_str, device_option_, &newblob);
    CHECK(blob != nullptr, "op: %s has a non-existing output blob",
          def.type().c_str(), output_str.c_str());
    outputs_.push_back(blob);
    if (newblob) {
      CHECK(k <= output_types.size(), "k=%u output_types.size()=%u %s", k, output_types.size(), def.name().c_str());
      blob->set_data_type(output_types[k]);
    }
  }
  LOG_DEBUG("name_=%s type_=%s outputs_.size()=%u",
            name_.c_str(), type_.c_str(), outputs_.size());
}

void OperatorBase::CheckSchema(OpSchema* schema) {
  // check input and output size
  size_t min_input = schema->min_input();
  size_t max_input = schema->max_input();
  BLAZE_CONDITION_THROW(min_input <= def_.input_size() && max_input >= def_.input_size(),
                        "min_input=", min_input, " max_input=", max_input,
                        " def_.input_size()=", def_.input_size());

  size_t min_output = schema->min_output();
  size_t max_output = schema->max_output();
  BLAZE_CONDITION_THROW(min_output <= def_.output_size() && max_output >= def_.output_size(),
                        "min_output=", min_output, " max_output=", max_output,
                        " def_.output_size()=", def_.output_size());
}

std::map<int, OperatorRegistry*>* GetDeviceTypeRegistry() {
  static std::map<int, OperatorRegistry*> g_device_type_registry;
  return &g_device_type_registry;
}

DEFINE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef,
    Workspace*);
REGISTER_DEVICE_TYPE(DeviceType::kCPU, CPUOperatorRegistry);

DEFINE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef,
    Workspace*);
REGISTER_DEVICE_TYPE(DeviceType::kCUDA, CUDAOperatorRegistry);

// Create opertor implementation
std::unique_ptr<OperatorBase> CreateOperator(const OperatorDef& def, Workspace* ws, int net_position) {
  const auto& op_type = def.type();
  const auto& device_type = def.device_option().device_type();

  auto* schema = OpSchemaRegistry::Schema(op_type);
  if (schema) {
    BLAZE_CONDITION_THROW(schema->Verify(def), "Verify schema failed.");
  } else {
    BLAZE_CONDITION_THROW("Schema for op_type ",
                          op_type,
                          " is not registered.");
  }

  BLAZE_CONDITION_THROW(GetDeviceTypeRegistry()->count(device_type),
                        "Device type ",
                        device_type,
                        " not registered.");
  OperatorRegistry* registry = GetDeviceTypeRegistry()->at(device_type);
  auto op = registry->Create(op_type, def, ws);
  BLAZE_CONDITION_THROW(op, "op_type=", op_type);
  op->set_net_position(net_position);
  return op;
}

}  // namespace blaze

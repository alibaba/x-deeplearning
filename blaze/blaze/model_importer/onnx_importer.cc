/*
 * \file onnx_importer.cc
 * \brief The onnx importer
 */
#include "blaze/model_importer/onnx_importer.h"

#include <sys/stat.h>

#include "blaze/common/exception.h"
#include "blaze/common/log.h"
#include "google/protobuf/message.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace blaze {

static std::string OnnxOpType2BlazeOpType(const std::string& onnx_op_type) {
  const static std::unordered_map<std::string, std::string> kReNameOperators = {
    { "Split", "Split" },
  };
  auto iter = kReNameOperators.find(onnx_op_type);
  if (iter != kReNameOperators.end()) {
    return iter->second;
  }
  return onnx_op_type;
}

static DataType TypeProto2DataType(onnx::TypeProto type) {
  switch (type.tensor_type().elem_type()) {
    case onnx::TensorProto::FLOAT:
      return kFloat;
    case onnx::TensorProto::DOUBLE:
      return kDouble;
    case onnx::TensorProto::FLOAT16:
      return kFloat16;
    default:
      BLAZE_THROW("Not supported data_type:",
                  type.tensor_type().elem_type(),
                  " please upgrade onnx_importer");
      break;
  }
  return kFloat;
}

OnnxAttributes::OnnxAttributes(const onnx::NodeProto& n) {
  for (const auto& attr : n.attribute()) {
    onnx_attrs_.emplace(attr.name(), &attr);
  }
}

google::protobuf::RepeatedPtrField<Argument> OnnxAttributes::OnnxAttr2BlazeArg(const std::string& op_type) {
  google::protobuf::RepeatedPtrField<Argument> args;
  for (const auto& kv : onnx_attrs_) {
    const auto& attr = *(kv.second);
    auto* arg = args.Add();
    arg->set_name(attr.name());
    CopyOnnxAttrValueToBlazeArg(arg, attr, op_type);
  }
  return args;
}

bool OnnxAttributes::ProcessAxesRelatedAttributes(Argument* arg, const onnx::AttributeProto& attr) {
  // Blaze now not support mutiple axes.
  if (attr.name() == "axes") {
    BLAZE_CONDITION_THROW(attr.ints_size() == 1, "attr.ints_size()=", attr.ints_size(), " name=", attr.name());
    arg->set_name("axis");
    arg->set_i(attr.ints(0));
  } else if (attr.name() == "starts") {
    BLAZE_CONDITION_THROW(attr.ints_size() == 1, "attr.ints_size()=", attr.ints_size(), " name=", attr.name());
    arg->set_name("start");
    arg->set_i(attr.ints(0));
  } else if (attr.name() == "ends") {
    BLAZE_CONDITION_THROW(attr.ints_size() == 1, "attr.ints_size()=", attr.ints_size(), " name=", attr.name());
    arg->set_name("end");
    arg->set_i(attr.ints(0));
  } else {
    return false;
  }
  return true;
}

void OnnxAttributes::CopyOnnxAttrValueToBlazeArg(Argument* arg, const onnx::AttributeProto& attr,
                                                 const std::string& op_type) {
  if (op_type == "Slice" || op_type == "ReduceSum") {
    // special process
    if (ProcessAxesRelatedAttributes(arg, attr)) return;
  }

  if (attr.has_f()) {
    arg->set_f(attr.f());
  } else if (attr.has_i()) {
    arg->set_i(attr.i());
  } else if (attr.has_s()) {
    arg->set_s(attr.s());
  } else if (attr.floats_size()) {
    arg->mutable_floats()->CopyFrom(attr.floats());
  } else if (attr.ints_size()) {
    arg->mutable_ints()->CopyFrom(attr.ints());
  } else if (attr.strings_size()) {
    arg->mutable_strings()->CopyFrom(attr.strings());
  } else {
    BLAZE_CONDITION_THROW("Not supported onnx attributes: ", attr.name());
  }
}

ONNXImporter::ONNXImporter() : ModelImporter() { }

void ONNXImporter::LoadModel(const char* model_conf, const char* model_data) {
  bool success = false;
  
  std::string content;
  success = ReadFileContent(model_conf, &content);
  CHECK_TRUE(success, "read model_conf=", model_conf, " failed");

  success = onnx_model_.ParseFromArray(content.c_str(), content.length());
  CHECK_TRUE(success, "parse onnx failed");

  success = ONNX2Blaze();
  CHECK_TRUE("onnx2blaze failed");

  LOG_DEBUG("net_def=%s", net_def_.DebugString().c_str());
}

bool ONNXImporter::ONNX2Blaze() {
  //net_def_.mutable_device_option()->set_device_type(device_type_);
  //net_def_.mutable_device_option()->set_device_id(device_id_);
  //net_def_.set_run_mode(run_mode_);

  std::unordered_set<std::string> initialized_inputs;
  for (const auto& tp : onnx_model_.graph().initializer()) {
    initialized_inputs.emplace(tp.name());
    if (!BuildConstantFillOp(tp)) {
      return false;
    }
    LOG_DEBUG("initializer name=%s", tp.name().c_str());
  }

  std::unordered_set<std::string> uninitialized_inputs;
  for (const auto& input : onnx_model_.graph().input()) {
    if (!initialized_inputs.count(input.name())) {
      uninitialized_inputs.emplace(input.name());
      auto vi = net_def_.add_external_input();
      vi->set_name(input.name());
      vi->set_dtype(TypeProto2DataType(input.type()));
      vi->set_doc_string(input.doc_string());
      LOG_DEBUG("input name=%s", input.name().c_str());
    }
  }
  for (const auto& output : onnx_model_.graph().output()) {
    auto vi = net_def_.add_external_output();
    vi->set_name(output.name());
    vi->set_dtype(TypeProto2DataType(output.type()));
    vi->set_doc_string(output.doc_string());
    LOG_DEBUG("output name=%s", output.name().c_str());
  }

  int index = 0;
  for (const auto& node : onnx_model_.graph().node()) {
    OnnxNode onnx_node(node);
    if (!ONNXNode2BlazeNode(&onnx_node, ++index)) {
      return false;
    }
  }
  RewriteOpDeviceOption();
  return true;
}

template <typename T>
bool TryConvertingTensorRawValues(const onnx::TensorProto& onnx_tensor,
                                  google::protobuf::RepeatedField<T>* field) {
  if (!onnx_tensor.has_raw_data()) {
    return false;
  }
  size_t raw_size = onnx_tensor.raw_data().size();
  CHECK(raw_size % sizeof(T) == 0, "raw_size=%u sizeof(T)=%u", raw_size, sizeof(T));

  size_t num_elements = raw_size / sizeof(T);
  const void* src_ptr = static_cast<const void*>(onnx_tensor.raw_data().data());
  field->Resize(num_elements, 0);
  void* target_ptr = static_cast<void*>(field->mutable_data());
  memcpy(target_ptr, src_ptr, raw_size);

  return true;
}

bool ONNXImporter::BuildConstantFillOp(const onnx::TensorProto& onnx_tensor) {
  auto* op = net_def_.add_op();
  op->set_name(onnx_tensor.name());
  op->add_output(onnx_tensor.name());
  op->set_type("ConstantFill");

  // set shape/dtype/value
  auto* shape = op->add_arg();
  shape->set_name("shape");
  for (const auto d : onnx_tensor.dims()) {
    shape->add_ints(d);
  }
  auto* dtype = op->add_arg();
  dtype->set_name("dtype");
  auto* value = op->add_arg();
  value->set_name("value");
  
  if (onnx_tensor.data_type() == onnx::TensorProto::FLOAT) {  
    dtype->set_i(kFloat);
    auto* floats = value->mutable_floats();
    if (!TryConvertingTensorRawValues<float>(onnx_tensor, floats)) {
      floats->CopyFrom(onnx_tensor.float_data());
    }
  } else if (onnx_tensor.data_type() == onnx::TensorProto::FLOAT16) {
    dtype->set_i(kFloat16);
    ::google::protobuf::RepeatedField<::google::protobuf::int32> tmp;
    const ::google::protobuf::RepeatedField<::google::protobuf::int32>* src = &tmp;
    if (!TryConvertingTensorRawValues<::google::protobuf::int32>(onnx_tensor, &tmp)) {
      src = &onnx_tensor.int32_data();
    }
    for (const auto i : *src) {
      value->add_ints(i);
    }
  } else if (onnx_tensor.data_type() == onnx::TensorProto::DOUBLE) {
    dtype->set_i(kDouble);
    google::protobuf::RepeatedField<double> tmp;
    const ::google::protobuf::RepeatedField<double>* src = &tmp;
    if (!TryConvertingTensorRawValues<double>(onnx_tensor, &tmp)) {
      src = &onnx_tensor.double_data();
    }
    for (const auto i : *src) {
      value->add_floats(i);
    }
  } else if (onnx_tensor.data_type() == onnx::TensorProto::INT32) {
    dtype->set_i(kInt32);
    google::protobuf::RepeatedField<int32_t> tmp;
    const ::google::protobuf::RepeatedField<int32_t>* src = &tmp;
    if (!TryConvertingTensorRawValues<int32_t>(onnx_tensor, &tmp)) {
      src = &onnx_tensor.int32_data();
    }
    for (const auto i : *src) {
      value->add_ints(i);
    }
  } else {
    BLAZE_THROW("Not supported data_type =",
                onnx_tensor.data_type(),
                " name = ", onnx_tensor.name().c_str(),
                " please upgrade onnx importer");
  }
  return true;
}

bool ONNXImporter::ONNXNode2BlazeNode(OnnxNode* onnx_node, int index) {
  auto* op = net_def_.add_op();

  const auto& node = onnx_node->node;
  op->mutable_input()->MergeFrom(node.input());
  op->mutable_output()->MergeFrom(node.output());
  op->set_name(node.name());
  op->set_type(OnnxOpType2BlazeOpType(node.op_type()));

  auto items = onnx_node->attributes.OnnxAttr2BlazeArg(op->type());
  op->mutable_arg()->MergeFrom(items);
  
  if (op->type() == "Reshape") {
    const std::string& name = op->input(1);
    reshape_constant_name_.insert(name);
    //LOG_INFO("Reshape name: %s", name.c_str());
  }
  LOG_INFO("index=%d onnx_node_type=%s blaze_node_type=%s name=%s",
           index, node.op_type().c_str(), op->type().c_str(), op->name().c_str());
  return true;
}

void ONNXImporter::RewriteOpDeviceOption() {
  for (size_t k = 0; k < net_def_.op_size(); ++k) {
    // The constant fill device option rewrite
    if (net_def_.op(k).type() == "ConstantFill") {
      if (reshape_constant_name_.count(net_def_.op(k).name())) {
        net_def_.mutable_op(k)->mutable_device_option()->set_device_type(kCPU);
        net_def_.mutable_op(k)->mutable_device_option()->set_device_id(0);
        //LOG_INFO("name %s deviceoption rewrite to CPU", net_def_.op(k).name().c_str());
      }
    }
  }
}

}  // namespace blaze

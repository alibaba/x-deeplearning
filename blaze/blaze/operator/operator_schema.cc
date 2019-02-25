/*
 * \file operator_schema.cc
 * \desc A class to record the schema of an op.
 *
 */
#include "blaze/operator/operator_schema.h"

#include <memory>

namespace blaze {

bool OpSchema::Verify(const OperatorDef& def) const {
  if (def.input_size() < min_input_ || def.input_size() > max_input_) {
    LOG_ERROR("Input size=%u not in range[min=%u, max=%u]. op=%s",
              def.input_size(), min_input_, max_input_, def.DebugString().c_str());
    return false;
  }
  if (!num_inputs_outputs_allowed(def.input_size(), def.output_size())) {
    LOG_ERROR("Input size=%u Output size=%u is not allowed!", def.input_size(), def.output_size());
    return false;
  }
  return true;
}

OpSchema& OpSchema::OpSchema::NumInputs(int n) {
  return NumInputs(n, n);
}

OpSchema& OpSchema::NumInputs(int min, int max) {
  min_input_ = min;
  max_input_ = max;
  return *this;
}

OpSchema& OpSchema::NumInputs(std::set<int>& allowed_input_nums) {
  return NumInputs([allowed_input_nums](int n)->bool {
                     return allowed_input_nums.count(n);
                   });
}

OpSchema& OpSchema::NumInputs(std::function<bool(int)> func) {
  num_inputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::NumOutputs(int n) {
  return NumOutputs(n, n);
}

OpSchema& OpSchema::NumOutputs(int min, int max) {
  min_output_ = min;
  max_output_ = max;
  return *this;
}

OpSchema& OpSchema::NumOutputs(std::set<int>& allowed_output_nums) {
  return NumOutputs([allowed_output_nums](int n)->bool {
                      return allowed_output_nums.count(n);
                    });
}

OpSchema& OpSchema::NumOutputs(std::function<bool(int)> func) {
  num_outputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::NumInputsOutputs(std::function<bool(int, int)> func) {
  num_inputs_outputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::OutputCalculator(std::function<int(int)> calc) {
  calculate_output_ = calc;
  return *this;
}

OpSchema& OpSchema::SameNumberOfOutput() {
  return OutputCalculator([](int n)->int { return n; });
}

OpSchema& OpSchema::AllowInplace(std::function<bool(int, int)> inplace) {
  inplace_allowed_ = inplace;
  return *this;
}

OpSchema& OpSchema::AllowInplace(std::set<std::pair<int, int>> inplace) {
  return AllowInplace([inplace](int in, int out)->bool {
                        return inplace.count(std::make_pair(in, out));
                      });
}

OpSchema& OpSchema::AllowOneToOneInplace() {
  return AllowInplace([](int in, int out) { return in == out; });
}

OpSchema& OpSchema::TypeInferenceFunction(TypeInferenceFunctionType function) {
  type_inference_function_ = function;
  return *this;
}

OpSchema& OpSchema::IdenticalType() {
  return TypeInferenceFunction([](
          const OperatorDef&, const std::vector<DataType>& input_types) {
        return std::vector<DataType>(input_types);
      }); 
}

OpSchema& OpSchema::IdenticalTypeOfInput(int idx) {
  return TypeInferenceFunction([idx](
          const OperatorDef& def, const std::vector<DataType>& input_types) {
        std::vector<DataType> out(def.output_size());
        for (size_t k = 0; k < def.output_size(); ++k) {
          out[k] = input_types[idx];
        }
        return out;
      });
}

OpSchema& OpSchema::ScalarType(DataType dt) {
  return TypeInferenceFunction([dt](
          const OperatorDef& def, const std::vector<DataType>&) {
        std::vector<DataType> out(def.output_size(), dt);
        return out;
      });
}

OpSchema& OpSchema::ShapeInferenceFunction(ShapeInferenceFunctionType function) {
  shape_inference_function_ = function;
  return *this;
}

OpSchema& OpSchema::IdenticalShape() {
  return ShapeInferenceFunction([](
          const OperatorDef&, const std::vector<TensorShape>& input_shapes) {
        return std::vector<TensorShape>(input_shapes);
      });
}

OpSchema& OpSchema::IdenticalShapeOfInput(int idx) {
  return ShapeInferenceFunction([idx](
          const OperatorDef& def, const std::vector<TensorShape>& input_shape) {
         std::vector<TensorShape> ret;
         for (size_t k = 0; k < def.output_size(); ++k) {
           ret.push_back(input_shape[idx]);
         }
         return ret;
      });
}

OpSchema& OpSchema::CostInferenceFunction(CostInferenceFunctionType function) {
  cost_inference_function_ =
      blaze::make_unique<CostInferenceFunctionType>(function);
  return *this;
}

OpSchema& OpSchema::SetDoc(const std::string& doc) {
  doc_ = doc;
  return *this;
}

OpSchema& OpSchema::Input(const int n, const char* name, const char* description) {
  if (input_desc_.size() <= n) {
    input_desc_.resize(n + 1);
  }
  input_desc_[n] = std::make_pair(name, description);
  return *this;
}

OpSchema& OpSchema::Output(const int n, const char* name, const char* description) {
  if (output_desc_.size() <= n) {
    output_desc_.resize(n + 1);
  }
  output_desc_[n] = std::make_pair(name, description);
  return *this;
}

int OpSchema::CalculateOutput(int num_input) const {
  if (min_output_ == max_output_) {
    return min_output_;
  } else if (calculate_output_) {
    return calculate_output_(num_input);
  } else {
    return -1;
  }
}

std::unordered_map<std::string, TensorShape> InferTensorShape(const NetDef& net_def) {
  std::unordered_map<std::string, TensorShape> infered_shape;
  for (const auto& external_input : net_def.external_input()) {
    const auto& name = external_input.name();
    const auto level = external_input.level();
    // structued-feature as follows:
    // level=2     level=1    level=0
    //   [] --------   [] --------   []
    //   [] --------   []     \  \   []
    //         \  \    []     \  \   []
    //                        \  \   []
    //
    //     level=1    level=0
    //       (indicator)     (indicator)
    //
    if (level >= 0) { // which is not indicator
      // check layer id must >= 0 and < 3.
      BLAZE_CONDITION_THROW(level >= 0 && level < 3, "level=", level);
      TensorShape tensor_shape;
      // NOTE: We deem the embedding's output is 2D dimentional.
      tensor_shape.add_dims(kL0BatchSize - level);
      tensor_shape.add_dims(kUnkownDim);
      infered_shape[name] = tensor_shape;
    }
  }

  for (const auto& op_def : net_def.op()) {
    std::vector<TensorShape> input_tensor_shape;
    for (const auto& name : op_def.input()) {
      const auto& iter = infered_shape.find(name);
      BLAZE_CONDITION_THROW(iter != infered_shape.end(),
                            "name=", name, "' shape is not infered");
      const auto& shape = iter->second;
      input_tensor_shape.push_back(shape);
    }
    const auto& type = op_def.type();
    const auto schema = OpSchemaRegistry::Schema(type);
    auto output_tensor_shape = schema->InferShape(op_def, input_tensor_shape);
    for (auto i = 0; i < op_def.output_size(); ++i) {
      const auto& name = op_def.output(i);
      infered_shape[name] = output_tensor_shape[i];
    }
  }
  return infered_shape;
}

BlazeMap<std::string, OpSchema>& OpSchemaRegistry::map() {
  static BlazeMap<std::string, OpSchema> map;
  return map;
}

}  // namespace blaze

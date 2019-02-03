/*
 * \file operator_schema.h
 * \desc A class to record the schema of an op.
 *
 * To register an OpSchema example:
 *
 *   OPERATOR_SCHEMA(name)
 *       .NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});
 */
#pragma once

#include <functional>
#include <initializer_list>
#include <limits>
#include <set>

#include "blaze/proto/blaze.pb.h"
#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/common/log.h"
#include "blaze/common/registry.h"
#include "blaze/common/types.h"
#include "blaze/operator/common_helper.h"
#include "blaze/proto/blaze.pb.h"

namespace blaze {

class OpSchema {
 public:
  OpSchema() : file_("unknown"), line_(0) { }
  OpSchema(const std::string& file, const int line) : file_(file), line_(line) { }

  // Verifies if an operator definition protobuf matches the pattern
  // specified in the schema.
  bool Verify(const OperatorDef& def) const;

  // The op input number config
  OpSchema& NumInputs(int n);
  OpSchema& NumInputs(int min, int max);
  OpSchema& NumInputs(std::set<int>& allowed_input_nums);
  OpSchema& NumInputs(std::function<bool(int)> func);

  // The op output number config
  OpSchema& NumOutputs(int n);
  OpSchema& NumOutputs(int min, int max);
  OpSchema& NumOutputs(std::set<int>& allowed_output_nums);
  OpSchema& NumOutputs(std::function<bool(int)> func);

  // Relationship between inputs and outputs
  OpSchema& NumInputsOutputs(std::function<bool(int, int)> func);

  // Set the function that can calculate the number of output based on the
  // number of input.
  OpSchema& OutputCalculator(std::function<int(int)> calc);

  // Set the number of ouputs to be the same as the number of inputs
  OpSchema& SameNumberOfOutput();

  // Sets the rule to allow in-place operation
  OpSchema& AllowInplace(std::function<bool(int, int)> inplace);
  OpSchema& AllowInplace(std::set<std::pair<int, int>> inplace);
  OpSchema& AllowOneToOneInplace();

  // Return true, if in_idx->out_idx allow inplaced.
  bool allow_inplace(int in_idx, int out_idx) {
    return inplace_allowed_(in_idx, out_idx);
  } 

  // Functions to deal with type inference.
  typedef std::function<
      std::vector<DataType>(const OperatorDef&,
                            const std::vector<DataType>)> TypeInferenceFunctionType;

  // Set the type inference functions.
  OpSchema& TypeInferenceFunction(TypeInferenceFunctionType function);

  OpSchema& IdenticalType();
  OpSchema& IdenticalTypeOfInput(int idx);
  OpSchema& ScalarType(DataType dt);

  // Infer type based on op schema
  inline std::vector<DataType> InferType(
      const OperatorDef& def,
      const std::vector<DataType>& input_type) const {
    return type_inference_function_(def, input_type);
  }

  // Functions to deal with shape inference
  typedef std::function<
      std::vector<TensorShape>(const OperatorDef&, const std::vector<TensorShape>&)> ShapeInferenceFunctionType;

  // Set the shape inference functions.
  OpSchema& ShapeInferenceFunction(ShapeInferenceFunctionType fuction);

  OpSchema& IdenticalShape();
  OpSchema& IdenticalShapeOfInput(int idx);

  // Infer shape based on op schema
  inline std::vector<TensorShape> InferShape(
      const OperatorDef& def,
      const std::vector<TensorShape>& input_shape) const {
    CHECK(shape_inference_function_ != nullptr,
          "shape inference function is not set for op %s", file_.c_str());
    return shape_inference_function_(def, input_shape);
  }

  //  A struct to store various cost information about an operator
  //  such as FLOPS, total memory use and parameters.
  struct Cost {
    uint64_t flops{ 0 };  // Floating point operations
    uint64_t bytes_read { 0 };  // Total memory read
    uint64_t bytes_written { 0 };  // Total memory written.
    uint64_t params_bytes { 0 } ;  // Memory size of parameters.

    Cost& operator+=(const Cost& cost) {
      flops += cost.flops;
      bytes_read += cost.bytes_read;
      bytes_written += cost.bytes_written;
      params_bytes += cost.params_bytes;
      return *this;
    }
    void Clear() {
      flops = 0;
      bytes_read = 0;
      bytes_written = 0;
      params_bytes = 0;
    }
  };

  typedef std::function<
      struct Cost(const OperatorDef&,
                  const std::vector<TensorShape>&,
                  const std::vector<DataType>&,
                  const std::vector<TensorShape>&,
                  const std::vector<DataType>&)> CostInferenceFunctionType;

  // Set the cost inference functions.
  OpSchema& CostInferenceFunction(CostInferenceFunctionType function);

  // Has cost inference function
  bool HasCostInferencFunction() const {
    return cost_inference_function_ != nullptr;
  }
  // Infer cost of the operator
  inline struct Cost InferCost(const OperatorDef& def,
                               const std::vector<TensorShape>& input_tensor_shape,
                               const std::vector<DataType>& input_type,
                               const std::vector<TensorShape>& output_tensor_shape,
                               const std::vector<DataType>& output_type) const {
    if (cost_inference_function_) {
      return (*cost_inference_function_)(def, input_tensor_shape, input_type,
                                         output_tensor_shape, output_type);
    } else {
      return Cost();
    }
  }
  
  // Set attribute for The op global attributes
  // Such as:
  //   kAttrIsElementWise
  template <typename T>
  OpSchema& SetAttr(const std::string& name, T value) {
    attrs_.SetAttr(name, value);
    return *this;
  }
  template <typename T>
  T GetAttr(const std::string& name, T default_value) {
    return attrs_.GetAttr(name, default_value);
  }

  // Functions to do documentation for the operator schema.
  OpSchema& SetDoc(const std::string& doc);

  // Set input and output
  OpSchema& Input(const int n, const char* name, const char* description);
  OpSchema& Output(const int n, const char* name, const char* description);

  // A function to allow one to get the number of outputs based on the number of
  // inputs, if this schema supports it.
  int CalculateOutput(int num_input) const;

  friend std::ostream operator<<(std::ostream& os, const OpSchema& schema);

  const std::string& file() const { return file_; }
  int line() const { return line_; }
  const char* doc() const { return doc_.c_str(); }
  
  int min_input() const { return min_input_; }
  int max_input() const { return max_input_; }
  int min_output() const { return min_output_; }
  int max_output() const { return max_output_; }
  bool num_inputs_allowed(int x) const { return num_inputs_allowed_(x); }
  bool num_outputs_allowed(int y) const { return num_outputs_allowed_(y); }
  bool num_inputs_outputs_allowed(int x, int y) const { return num_inputs_outputs_allowed_(x, y); }
  
  int inf() const { return std::numeric_limits<int>::max(); }
  
  const std::vector<std::pair<const char*, const char*>>& input_desc() const {
    return input_desc_;
  }
  const std::vector<std::pair<const char*, const char*>>& output_desc() const {
    return output_desc_;
  }

 protected:
  std::string file_;
  std::string doc_;
  std::vector<std::pair<const char*, const char*>> input_desc_{};
  std::vector<std::pair<const char*, const char*>> output_desc_{};
  int line_ = 0;
  int min_input_ = 0;
  int max_input_ = std::numeric_limits<int>::max();
  int min_output_ = 0;
  int max_output_ = std::numeric_limits<int>::max();
  
  std::function<bool(int)> num_inputs_allowed_ = [](int) { return true; };
  std::function<bool(int)> num_outputs_allowed_ = [](int) { return true; };
  std::function<bool(int, int)> num_inputs_outputs_allowed_ = [](int, int) { return true; };
  std::function<int(int)> calculate_output_;
  
  std::function<bool(int, int)> inplace_allowed_ = [](int, int) { return false; };
  
  TypeInferenceFunctionType type_inference_function_ =
      [](const OperatorDef& def, const std::vector<DataType>& input_type) {
        std::vector<DataType> out;
        for (int i = 0; i < def.output_size(); i++) {
          out.push_back(input_type[0]);
        }
        return out;
      };
  std::unique_ptr<CostInferenceFunctionType> cost_inference_function_ = nullptr;
  ShapeInferenceFunctionType shape_inference_function_ = nullptr;

  AttrMap attrs_;
};

template <int OpsPerElement>
OpSchema::Cost ElementWiseCostInference(const OperatorDef&,
                                        const std::vector<TensorShape>& input_shape,
                                        const std::vector<DataType>& input_type,
                                        const std::vector<TensorShape>& output_shape,
                                        const std::vector<DataType>& output_type) {
  OpSchema::Cost c;
  size_t elem_read = 0;
  for (int i = 0; i < input_shape.size(); ++i) {
    elem_read += NElemFromDim(input_shape[i]) * DataTypeSize(input_type[i]);
  }
  c.bytes_read = elem_read;
  size_t elem_write = 0;
  size_t out_elem_num = 0;
  for (int i = 0; i < output_shape.size(); ++i) {
    out_elem_num += NElemFromDim(output_shape[i]);
    elem_write += NElemFromDim(output_shape[i]) * DataTypeSize(output_type[i]);
  }
  c.bytes_written = elem_write;
  c.flops = out_elem_num * OpsPerElement;
  return c;
}

enum {
  // This is a unkown dim.
  kUnkownDim = -1,
  // The leaf feature
  kL0BatchSize = std::numeric_limits<int32_t>::max(),
  // The middle feature
  kL1BatchSize = kL0BatchSize - 1,
  // The top feature
  kL2BatchSize = kL1BatchSize - 1,
};

// Infer the tensor shape of the net_def.
extern std::unordered_map<std::string, TensorShape> InferTensorShape(const NetDef& net_def);

// A registry to hold all the operator schemas
class OpSchemaRegistry {
 public:
  static OpSchema& NewSchema(const std::string& key, const std::string& file, const int line) {
    auto& m = map();
    auto it = m.find(key);
    if (it != m.end()) {
      const auto& schema = it->second;
      BLAZE_THROW("Trying to register schema with name ",
                  key,
                  " from file ",
                  file,
                  " line ",
                  line,
                  ", but it is already registered from file ",
                  schema.file(),
                  " line ",
                  schema.line());
    }
    m.emplace(std::make_pair(key, OpSchema(file, line)));
    return m[key];
  }

  static OpSchema* Schema(const std::string& key) {
    auto& m = map();
    auto it = m.find(key);
    if (it != m.end()) {
      return &it->second;
    } else {
      return nullptr;
    }
  }

 private:
  OpSchemaRegistry() = delete;

  static BlazeMap<std::string, OpSchema>& map();
};

#define OPERATOR_SCHEMA(name)                                       \
    void PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name() { };               \
    static OpSchema* ANONYMOUS_VARIABLE(name) UNUSED =              \
       &OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)

}  // namespace blaze

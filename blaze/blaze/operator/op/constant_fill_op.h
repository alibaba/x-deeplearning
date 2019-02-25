/*
 * \file constant_fill_op.h 
 * \brief The constant fill operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

// NOTE: Don't change shape/dtype/value key names, because it is used
// in fusion pattern, such as GEMM/GEMM fusion.
template <class Context>
class ConstantFillOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  ConstantFillOp(const OperatorDef& def, Workspace* ws) :
      Operator<Context>(def, ws, false),  // We create constant blob by hand.
    shape_(OperatorBase::GetRepeatedArgument<TIndex>("shape")) {
    DataType dtype = static_cast<DataType>(OperatorBase::GetSingleArgument<int>("dtype", kFloat));
    
    for (const std::string& output_str : def.output()) {
      Blob* blob = ws->CreateConstantBlob(output_str, this->device_option_);
      this->outputs_.push_back(blob);
    }
 
    auto* output = Operator<Context>::Output(0);
    output->set_data_type(dtype);
    output->Reshape(shape_);

    CONSTANT_FILL_TYPE_SWITCH(dtype, DType, {
      FillWithType<DType>(output);
    });
  }

  bool RunOnDevice() override { return true; }

 protected:
  template <typename DType>
  void FillWithType(Blob* output) {
    const std::vector<DType> value = OperatorBase::GetRepeatedArgument<DType>("value");
    BLAZE_CONDITION_THROW(value.size() == output->size(),
                          "value.size()=",
                          value.size(),
                          " output->size()=",
                          output->size(), this->def_.DebugString());
    DType* dst = output->as<DType>();
    CopyData(dst, value.data(), sizeof(DType) * output->size());
  }
  // Copydata implementation
  void CopyData(void* dst, const void* src, size_t size);

  std::vector<TIndex> shape_;
};

}  // namespace blaze


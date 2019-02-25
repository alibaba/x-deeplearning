/*!
 * \file udf_processor.h
 * \desc udf processor for embedding result
 */
#pragma once

#include "blaze/store/sparse_puller.h"
#include "blaze/common/exception.h"

namespace blaze {
namespace store {

// basic definition of udf process
template <typename DictValueType, typename ValueType>
class UdfProcessor {
 public:
  virtual bool InitProcess(int dim,
                           const SparsePullerInput::Param& param,
                           ValueType* out) {
    return true;
  }

  virtual bool ElementProcess(ValueType value,
                              const DictValueType* weights,
                              int dim,
                              size_t index,
                              size_t size,
                              const SparsePullerInput::Param& param,
                              ValueType* out) {
    return true;
  }

  virtual bool ReduceProcess(int dim,
                             size_t size,
                             const SparsePullerInput::Param& param,
                             ValueType* out) {
    return true;
  }
};

// ksum processor
template <typename DictValueType, typename ValueType>
class KSumProcessor : public UdfProcessor<DictValueType, ValueType> {
 public:
  static KSumProcessor<DictValueType, ValueType>* GetInstance() {
    static KSumProcessor<DictValueType, ValueType> instance;
    return &instance;
  }

  bool InitProcess(int dim,
                   const SparsePullerInput::Param& param,
                   ValueType* out) override {
    memset(out, 0, dim * sizeof(ValueType));
    return true;
  }

  bool ElementProcess(ValueType value,
                      const DictValueType* weights,
                      int dim,
                      size_t index,
                      size_t size,
                      const SparsePullerInput::Param& param,
                      ValueType* out) override {
    auto current_out = out;
    for (auto i = 0; i < dim; ++i) {
      *(current_out) = *(current_out) + value * (ValueType)(weights[i]);
      ++current_out;
    }
    return true;
  }
};

// kavg processor
template <typename DictValueType, typename ValueType>
class KAvgProcessor : public KSumProcessor<DictValueType, ValueType> {
 public:
  static KAvgProcessor<DictValueType, ValueType>* GetInstance() {
    static KAvgProcessor<DictValueType, ValueType> instance;
    return &instance;
  }

  bool ReduceProcess(int dim,
                     size_t size,
                     const SparsePullerInput::Param& param,
                     ValueType* out) override {
    if (size == 0) return true;
    for (auto i = 0; i < dim; ++i) {
      out[i] = out[i] * ValueType(1.0 / size);
    }
    return true;
  }
};

// kassign processor
template <typename DictValueType, typename ValueType>
class KAssignProcessor : public UdfProcessor<DictValueType, ValueType> {
 public:
  static KAssignProcessor<DictValueType, ValueType>* GetInstance() {
    static KAssignProcessor<DictValueType, ValueType> instance;
    return &instance;
  }

  bool InitProcess(int dim,
                   const SparsePullerInput::Param& param,
                   ValueType* out) override {
    memset(out, 0, dim * param.trunc_num * sizeof(ValueType));
    return true;
  }

  bool ElementProcess(ValueType value,
                      const DictValueType* weights,
                      int dim,
                      size_t index,
                      size_t size,
                      const SparsePullerInput::Param& param,
                      ValueType* out) override {
    if (param.trunc_direction == kOrder && index >= param.trunc_num) return true;
    if (param.trunc_direction == kReverse && (size - index) > param.trunc_num) return true;

    auto current_out = out;
    if (param.trunc_direction == kOrder) {
      current_out += dim * index;
    } else {
      current_out += dim * (param.trunc_num - (size - index));
    }
    for (auto i = 0; i < dim; ++i) {
      *(current_out) = value * (ValueType)(weights[i]);
      ++current_out;
    }

    return true;
  }
};

// udf processor factory
template <typename DictValueType, typename ValueType>
class UdfProcessorFactory {
 public:
  static UdfProcessor<DictValueType, ValueType>* Create(UDFType udf_type) {
    switch (udf_type) {
      case UDFType::kSum:
        return KSumProcessor<DictValueType, ValueType>::GetInstance();
      case UDFType::kAvg:
        return KAvgProcessor<DictValueType, ValueType>::GetInstance();
      case UDFType::kAssign:
        return KAssignProcessor<DictValueType, ValueType>::GetInstance();
      default:
        BLAZE_THROW("Unsupported udf type:", udf_type);
    }
  }
};

}  // namespace store
}  // namespace blaze


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


#include "xdl/data_io/op/feature_op/expr/expr_node.h"

#include "xdl/data_io/op/feature_op/expr/internal_feature.h"
#include "xdl/data_io/op/feature_op/feature_util.h"
#include "xdl/data_io/op/feature_op/multi_feature_op/multi_feature_op_factory.h"
#include "xdl/data_io/op/feature_op/single_feature_op/single_feature_op_factory.h"
#include "xdl/data_io/op/feature_op/source_feature_op/source_feature_op_factory.h"
#include "xdl/proto/sample.pb.h"

namespace xdl {
namespace io {

void ExprNode::clear() {
  if (internal) {
    InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    feature->clear();
  }
}

void ExprNode::reserve(int capacity) {
  if (internal) {
    InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    feature->reserve(capacity);
  }
}

int ExprNode::capacity() const {
  if (internal) {
    InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    return feature->capacity();
  } else {
    return values_size();
  }
}

void ExprNode::add(int64_t key, float value) {
  if (internal) {
    InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    feature->push_back(key, value);
  } else {
    Feature *feature = reinterpret_cast<Feature *>(result);
    FeatureValue *feature_value = feature->add_values();
    if (key > 0)  feature_value->set_key(key);
    feature_value->set_value(value);
  }
}

void ExprNode::get(int index, int64_t &key, float &value) const {
  if (internal) {
    const InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    const InternalValue &internal_value = feature->values(index);
    key = internal_value.key();
    value = internal_value.value();
  } else {
    const Feature *feature = reinterpret_cast<Feature *>(result);
    const FeatureValue &feature_value = feature->values(index);
    key = FeatureUtil::GetKey(feature_value);
    value = FeatureUtil::GetValue(feature_value);
  }
}

int64_t ExprNode::key(int index) const {
  if (internal) {
    const InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    return feature->values(index).key();
  } else {
    const Feature *feature = reinterpret_cast<Feature *>(result);
    return FeatureUtil::GetKey(feature->values(index));
  }
}

float ExprNode::value(int index) const {
  if (internal) {
    const InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    return feature->values(index).value();
  } else {
    const Feature *feature = reinterpret_cast<Feature *>(result);
    return FeatureUtil::GetValue(feature->values(index));
  }
}

int ExprNode::values_size() const {
  if (internal) {
    const InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    return values_size<InternalFeature>(feature);
  } else {
    const Feature *feature = reinterpret_cast<Feature *>(result);
    return values_size<Feature>(feature);
  }
}

int ExprNode::BinarySearch(int64_t key, float &value) const {
  if (internal) {
    const InternalFeature *feature = reinterpret_cast<InternalFeature *>(result);
    return BinarySearch<InternalFeature>(feature, key, value);
  } else {
    const Feature *feature = reinterpret_cast<Feature *>(result);
    return BinarySearch<Feature>(feature, key, value);
  }
}

void ExprNode::InitResult() {
  if (internal)  result = new InternalFeature();
  else if (output)  result = new Feature();
  else  result = nullptr;
}

void ExprNode::ReleaseResult() {
  if (result != nullptr) {
    if (internal)  delete reinterpret_cast<InternalFeature *>(result);
    else if (output)  delete reinterpret_cast<Feature *>(result);
  }
}

void ExprNode::ReleaseOp() {
  if (op != nullptr) {
    switch (type) {
     case FeaOpType::kSourceFeatureOp:
      SourceFeatureOpFactory::Release(reinterpret_cast<SourceFeatureOp *>(op));
      break;
     case FeaOpType::kSingleFeatureOp:
      SingleFeatureOpFactory::Release(reinterpret_cast<SingleFeatureOp *>(op));
      break;
     case FeaOpType::kMultiFeatureOp:
      MultiFeatureOpFactory::Release(reinterpret_cast<MultiFeatureOp *>(op));
      break;
     default:
      return;
    }
    op = nullptr;
  }
}

}  // namespace io
}  // namespace xdl
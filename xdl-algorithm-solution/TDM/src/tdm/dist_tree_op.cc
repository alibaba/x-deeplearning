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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

#include "tair_store.h"
#include "dist_tree.h"
#include "selector.h"

namespace util {

REGISTER_OP("DistTree")
    .Attr("store_config: string")
    .Attr("branch: int")
    .Attr("key_prefix: string")
    .Attr("strategy: string")
    .Attr("strategy_config: string = ''")
    .Attr("strategy_so: string = ''")
    .Input("item_ids: int64")
    .Input("features: int64")
    .Input("numbers: int32")
    .Output("o_item_ids: int64")
    .Output("o_scores: float");

class DistTreeOp: public tensorflow::OpKernel {
 public:
  explicit DistTreeOp(tensorflow::OpKernelConstruction* context);
  void Compute(tensorflow::OpKernelContext* context) override;

 private:
  TairStore store_;
  DistTree dist_tree_;
  Selector* selector_;
  static const int kPersistLevel = 10;
};

DistTreeOp::DistTreeOp(tensorflow::OpKernelConstruction* context)
    : OpKernel(context) {
  std::string store_config;
  int branch = 2;
  std::string key_prefix;
  std::string strategy;
  std::string strategy_so;
  std::string strategy_config;

  OP_REQUIRES_OK(context,
                 context->GetAttr("store_config", &store_config));
  OP_REQUIRES_OK(context,
                 context->GetAttr("branch", &branch));
  OP_REQUIRES_OK(context,
                 context->GetAttr("key_prefix", &key_prefix));
  OP_REQUIRES_OK(context,
                 context->GetAttr("strategy", &strategy));
  OP_REQUIRES_OK(context,
                 context->GetAttr("strategy_so", &strategy_so));

  OP_REQUIRES_OK(context,
                 context->GetAttr("strategy_config", &strategy_config));

  if (!store_.Init(store_config)) {
    printf("[ERROR] Store init failed, config: %s\n", store_config.c_str());
    return;
  }

  dist_tree_.set_branch(branch);
  dist_tree_.set_store(&store_);
  dist_tree_.set_key_prefix(key_prefix);
  if (!dist_tree_.Load()) {
    printf("[ERROR] DistTree load failed!");
    return;
  }
  dist_tree_.Persist(kPersistLevel);

  if (!SelectorMapper::LoadSelector(strategy_so)) {
    printf("[ERROR] So file [%s] not exists!\n", strategy_so.c_str());
  }

  selector_ = SelectorMapper::GetSelector(strategy);
  if (selector_ == NULL) {
    printf("[ERROR] Strategy [%s] not exists!\n", strategy.c_str());
    return;
  }

  selector_->set_dist_tree(&dist_tree_);
  if (!strategy_config.empty() && !selector_->Init(strategy_config)) {
    printf("[ERROR] Strategy [%s] init failed, config: %s\n", strategy.c_str(), strategy_config.c_str());
    delete selector_;
    selector_ = NULL;
  }
}

void DistTreeOp::Compute(tensorflow::OpKernelContext* context) {
  if (selector_ == NULL) {
    printf("[ERROR] DistTreeOp Initialized failed\n");
    return;
  }

  auto item_ids = context->input(0).flat<tensorflow::int64>();
  auto features = context->input(1);
  auto numbers = context->input(2).flat<tensorflow::int32>();

  if (features.dims() != 2) {
    return;
  }

  int item_numer = item_ids.size();
  int sample_num = 1;
  for (int i = 0; i < numbers.size(); ++i) {
    sample_num += numbers(i);
  }

  // Create an output tensor
  tensorflow::Tensor* o_item_ids = NULL;
  tensorflow::Tensor* o_scores = NULL;

  tensorflow::TensorShape shape({item_numer, sample_num});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, shape, &o_item_ids));
  OP_REQUIRES_OK(context,
                 context->allocate_output(1, shape, &o_scores));

  std::vector<int64_t> v_o_item_ids(item_numer * sample_num);
  std::vector<float> v_o_scores(item_numer * sample_num);
  std::vector<int64_t> v_i_item_ids(item_numer);
  std::vector<std::vector<int64_t> > v_i_features(features.dim_size(0));
  std::vector<int> v_i_numbers(numbers.size());
  for (int i = 0; i < item_numer; ++i) {
    v_i_item_ids[i] = item_ids(i);
  }

  int dim1 = features.dim_size(1);
  auto f = features.flat<tensorflow::int64>();
  for (size_t i = 0; i < v_i_features.size(); ++i) {
    v_i_features[i].resize(dim1);
    for (int j = 0; j < dim1; ++j) {
      v_i_features[i][j] = f(i * dim1 + j);
    }
  }

  for (int i = 0; i < numbers.size(); ++i) {
    v_i_numbers[i] = numbers(i);
  }

  selector_->Select(v_i_item_ids, v_i_features, v_i_numbers,
                    v_o_item_ids.data(), v_o_scores.data());

  auto o_item_ids_flat = o_item_ids->flat<tensorflow::int64>();
  auto o_scores_flat = o_scores->flat<float>();
  for (size_t i = 0; i < v_o_item_ids.size(); ++i) {
    o_item_ids_flat(i) = v_o_item_ids[i];
    o_scores_flat(i) = v_o_scores[i];
  }
}

REGISTER_KERNEL_BUILDER(
    Name("DistTree").Device(tensorflow::DEVICE_CPU), DistTreeOp);

}  // namespace util

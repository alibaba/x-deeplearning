/*
 * \file gru_op.h
 * \brief The gru operation
 */
#pragma once

#include <memory>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "blaze/operator/operator.h"
#include "blaze/common/cuda_helpers.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

const int gru_weights_per_thread = 8;

const int gru_threads_per_block = 256;

template <typename DType>
struct GRUParam {
  DType* x;
  DType* h2h;
  DType* h2h_bias;
  DType* i2h;
  DType* i2h_bias;
  DType* y;
  DType* preact;
  DType* act;
  unsigned int* finished;
  TIndex batch_size;
  TIndex round;
  TIndex elts;
};

template <class Context>
class GRUOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  GRUOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) {
    mask_ = OperatorBase::GetSingleArgument<bool>("mask", true);
    preact_ = std::unique_ptr<Blob>(new Blob(this->device_option_));
    act_ = std::unique_ptr<Blob>(new Blob(this->device_option_));
    finished_ = std::unique_ptr<Blob>(new Blob(this->device_option_));
    from_deepnet_ =
      OperatorBase::GetSingleArgument<bool>("from_deepnet", false);
#ifdef USE_CUDA
    device_prop_ =
      (this->device_option_.device_type() == DeviceType::kCUDA
       ? GetDeviceProp(this->device_option_.device_id())
       : nullptr);
#endif
  }

  bool RunOnDevice() override;

 protected:

  template <typename DType>
  void Setup(GRUParam<DType>* param) {
    if (from_deepnet_) {
      auto x = Input(0);
      auto y = Output(0);
      auto h2h = Input(1);

      const std::vector<TIndex>& x_shape = x->shape();
      BLAZE_CONDITION_THROW(x_shape.size() <= 3,
                            "x_shape.size()=", x_shape.size());

      auto batch_size = x_shape.size() < 3 ? 1 : x_shape[0];
      auto round = x_shape[x_shape.size() - 2];
      auto elts = h2h->shape()[0];

      // N.B. The following initializations are tricky because we need
      // to support both CPU and GPU implementations. It is sufficient
      // to prepare large enough space here.
      std::vector<TIndex> shape { batch_size, round, elts };
      y->Reshape(shape);
      shape[2] *= 3;
      preact_->Reshape(shape);
      act_->Reshape({
        batch_size,
          elts * (alignN(elts, gru_weights_per_thread)
                  / gru_weights_per_thread) * 3 * 2 });
      finished_->Reshape({ round });

      // x, h2hweight, i2hweight, h2hBias, i2hbias, preact
      param->x = this->Input(0)->template as<DType>();
      param->h2h = this->Input(1)->template as<DType>();
      param->i2h = this->Input(2)->template as<DType>();
      param->h2h_bias = this->Input(3)->template as<DType>();
      param->i2h_bias = this->Input(4)->template as<DType>();
      param->y = this->Output(0)->template as<DType>();
      param->preact = preact_->template as<DType>();
      param->act = act_->template as<DType>();
      param->finished = finished_->template as<unsigned int>();
      param->batch_size = batch_size;
      param->round = round;
      param->elts = elts;
    } else {
      auto x = Input(0);
      auto y = Output(1);
      auto i2h = Input(1);
      auto h2h = Input(2);
      auto bias = Input(3);

      const std::vector<TIndex>& x_shape = x->shape();
      BLAZE_CONDITION_THROW(x_shape.size() <= 3,
                            "x_shape.size()=", x_shape.size());

      auto batch_size = x_shape.size() < 3 ? 1 : x_shape[0];
      auto round = x_shape[x_shape.size() - 2];
      auto elts = h2h->shape()[0];

      // N.B. The following initializations are tricky because we need
      // to support both CPU and GPU implementations. It is sufficient
      // to prepare large enough space here.
      std::vector<TIndex> shape { batch_size, round, elts };
      y->Reshape(shape);
      shape[2] *= 3;
      preact_->Reshape(shape);
      act_->Reshape({
        batch_size,
          elts * (alignN(elts, gru_weights_per_thread)
                  / gru_weights_per_thread) * 3 * 2});
      finished_->Reshape({ round });

      param->x = x->template as<DType>();
      param->h2h = h2h->template as<DType>();
      param->i2h = i2h->template as<DType>();
      param->h2h_bias = &bias->template as<DType>()[bias->size() / 2];
      param->i2h_bias = bias->template as<DType>();
      param->y = y->template as<DType>();
      param->preact = preact_->template as<DType>();
      param->act = act_->template as<DType>();
      param->finished = finished_->template as<unsigned int>();
      param->batch_size = batch_size;
      param->round = round;
      param->elts = elts;
    }
  }

  bool mask_;
  bool from_deepnet_;
  std::unique_ptr<Blob> preact_;
  std::unique_ptr<Blob> act_;
  std::unique_ptr<Blob> finished_;
#ifdef USE_CUDA
  cudaDeviceProp* device_prop_;
#endif
};

}  // namespace blaze

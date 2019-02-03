/*
 * \file multi_slice_op.h 
 * \brief The multi slice operation
 */
#pragma once

#include <vector>

#include "blaze/common/exception.h"
#include "blaze/common/types.h"
#include "blaze/operator/common_helper.h"
#include "blaze/operator/operator.h"
#include "blaze/math/memory.h"

namespace blaze {

struct MultiSliceItem {
  size_t start;
  size_t end;
};

template <typename DType>
struct MultiSliceParam {
  DType* x;
  size_t x_row;
  size_t x_col;

  size_t concat_dim_;

  MultiSliceItem slice_item[kMaxInputSize];
};

template <class Context>
class MultiSliceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  MultiSliceOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace), use_memcpy2d_(false) {
    concat_dim_ = OperatorBase::GetSingleArgument<size_t>("concat_dim", 1);
    offset_ = OperatorBase::GetRepeatedArgument<size_t>("offset");
    std::vector<size_t> shape = OperatorBase::GetRepeatedArgument<size_t>("shape");
    BLAZE_CONDITION_THROW(shape.size() % offset_.size() == 0,
                          "shape.size()=", shape.size(),
                          " offset_.size()=", offset_.size());
    for (size_t i = 0; i < shape.size(); ) {
      std::vector<size_t> slice_shape;
      for (int k = 0; k < shape.size() / offset_.size(); ++k) {
        slice_shape.push_back(shape[i]);
        ++i;
      }
      shape_.push_back(slice_shape);
    }
    for (size_t k = 1; k < shape_.size(); ++k) {
      BLAZE_CONDITION_THROW(shape_[k - 1].size() == shape_[k].size(),
                            "shape_[", k - 1, "].size()=", shape_[k - 1].size(),
                            " shape_[", k, "].size()=", shape_[k].size());
      for (size_t z = 0; z < shape_[k - 1].size(); ++z) {
        if (z != concat_dim_) {
          BLAZE_CONDITION_THROW(shape_[k - 1][z] == shape_[k][z],
                                "shape_[", k - 1, "][", z , "]=", shape_[k - 1][z],
                                " shape_[", k, "][", z, "]=", shape_[k][z]);
        }
      }
    }
    CalcConcatHeight();
    CalcConcatStep();
    CalcSliceParam();
    
    tmp_blob_.reset(new Blob(this->device_option()));
    tmp_blob_->set_data_type(static_cast<DataType>(DataType::kInt64));
  }

  bool RunOnDevice() override;
 
 protected:
  void Memcpy2D(void* dst, size_t dpitch,
    const void* src, size_t spitch,
    size_t width, size_t height) const;

  void CalcConcatHeight() {
    concat_height_ = 1;
    if (shape_.size() > 0) {
      BLAZE_CONDITION_THROW(concat_dim_ < shape_[0].size(),
          "concat_dim_: ", concat_dim_,
          "; slice_shape dim: ", shape_[0].size());
      for (int i = 0; i < concat_dim_; ++i) {
        concat_height_ *= shape_[0][i]; 
      }
      slice_count_ = shape_.size() * concat_height_;
    }
  }

  void CalcConcatStep() {
    for (int i = 0; i < shape_.size(); ++i) {
      size_t cur_concat_step = 1;
      for (int j = concat_dim_; j < shape_[i].size(); ++j) {
        cur_concat_step *= shape_[i][j];
      }
      concat_step_.push_back(cur_concat_step);
    }
  }

  void CalcSliceParam() {
    size_t dst_idx = 0;
    for (int i = 0; i < concat_height_; ++i) {
      for (int j = 0; j < shape_.size(); ++j) {
        size_t step_size = concat_step_[j]; 
        size_t src_idx = offset_[j] + i * step_size;
        slice_param_.emplace_back(src_idx, dst_idx, step_size);   
        dst_idx += step_size; 
      }
    }
    total_step_ = dst_idx;
    const int CONCAT_STEP_BOUNDARY = 256; 
    if (total_step_ / slice_count_ > CONCAT_STEP_BOUNDARY) {
      use_memcpy2d_ = true; 
    }
  }

  template <typename T>
  void Setup(MultiSliceParam<T>* param) {
    Blob* x = this->Input(0);
    Blob* y = this->Output(0);

    // Step1: Reshape y.
    TIndex batch_size = x->shape()[0];
    std::vector<TIndex> shape = shape_[0];
    shape[concat_dim_] = 0;
    for (const auto& slice : shape_) shape[concat_dim_] += slice[concat_dim_];

    shape.insert(shape.begin(), batch_size);
    y->Reshape(shape);
  }

  template <typename DType>
  void SliceMemcpy2D() {
    Blob* x = this->Input(0);
    Blob* y = this->Output(0);
    TIndex batch_size = y->shape()[0];
    DType* dst = y->as<DType>();
    DType* src = x->as<DType>();
    TIndex x_stride = x->size(1, x->shape().size()); 
    for (int i = 0; i < slice_count_; ++i) {
      Memcpy2D(dst + slice_param_[i].dst_idx,
        total_step_ * sizeof(DType),
        src + slice_param_[i].src_idx,
        x_stride * sizeof(DType),
        slice_param_[i].step_size * sizeof(DType),
        batch_size); 
    }
  }

  template <typename DType>
  void MultiSliceMemcpy() {
    Blob* x = this->Input(0);
    Blob* y = this->Output(0);
    TIndex batch_size = y->shape()[0];
    DType* dst = y->as<DType>();
    DType* src = x->as<DType>();
    TIndex x_stride = x->size(1, x->shape().size());
    // create extra blob
    std::vector<TIndex> tmp_shape(1, sizeof(SliceParam) * slice_param_.size()); 
    //std::unique_ptr<Blob> tmp_blob(new Blob(this->device_option_,
    //      tmp_shape, DataType::kInt64));
    tmp_blob_->Reshape(tmp_shape);
    SliceParam* tmp_buf = tmp_blob_->as<SliceParam>();
    Memcpy<Context>(tmp_buf, slice_param_.data(),
        sizeof(SliceParam) * slice_param_.size(),
        &(this->context()));   

    SliceMemcpy<DType, Context>(dst, total_step_,
        src, x_stride,
        tmp_buf,
        slice_count_, batch_size,
        &(this->context()));
  }  

  std::unique_ptr<Blob> tmp_blob_;
  size_t concat_dim_;
  size_t concat_height_;
  size_t slice_count_;
  size_t total_step_;
  bool use_memcpy2d_;
  std::vector<size_t> offset_;
  std::vector<std::vector<size_t>> shape_;
  std::vector<size_t> concat_step_;
  std::vector<SliceParam> slice_param_;
};

}  // namespace blaze

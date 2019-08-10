/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"

namespace xdl {

class SplitOp : public xdl::OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("dtype", &dtype_));
    XDL_CHECK_STATUS(ctx->GetAttr("axis", &axis_));
    XDL_CHECK_STATUS(ctx->GetAttr("num", &num_));
    if (num_ <= 0) {
      return Status::ArgumentError("Split num should greater than 0");
    }
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor value, num_or_size_splits;
    XDL_CHECK_STATUS(ctx->GetInput(0, &value));
    XDL_CHECK_STATUS(ctx->GetInput(1, &num_or_size_splits));
    int64_t real_axis;
    if (axis_ < 0) {
      if (-axis_ > value.Shape().Size()) {
        return Status::ArgumentError("Split axis_ should be [-value_dims, value_dims)");
      }
      real_axis = value.Shape().Size() + axis_;
    } else {
      if (axis_ >= value.Shape().Size()) {
        return Status::ArgumentError("Split axis_ should be [-value_dims, value_dims)");
      }
      real_axis = axis_;
    }
    int64_t dim = value.Shape()[real_axis];
    std::vector<int64_t> dims;
    if (num_or_size_splits.Shape().Size() == 0) {
      if (num_or_size_splits.Raw<int64_t>()[0] != num_) {
        return Status::ArgumentError("Split num_or_size_splits should equal to num");
      }
      if (dim % num_ != 0) {
        return Status::ArgumentError("Split dim should evenly divide num");
      }
      for (int64_t i = 0; i < num_; i++) {
        dims.push_back(dim / num_);
      }
    } else if (num_or_size_splits.Shape().Size() == 1) {
      if (num_or_size_splits.Shape()[0] != num_) {
        return Status::ArgumentError("Split num_or_size_splits should be scalar or [num_] shaped");
      }
      int64_t sum = 0;
      for (int64_t i = 0; i < num_; i++) {
        sum += num_or_size_splits.Raw<int64_t>()[i];
        dims.push_back(num_or_size_splits.Raw<int64_t>()[i]);
      }
      if (sum != dim) {
        return Status::ArgumentError("Split num_or_size_splits is not matched");
      }
    } else {
        return Status::ArgumentError("Split num_or_size_splits should be scalar or 1-D");
    }
    int64_t slice_size = SizeOfType(dtype_), repeat = 1;
    for (int64_t i = real_axis + 1; i < value.Shape().Size(); i++) {
      slice_size *= value.Shape()[i];
    }
    for (int64_t i = 0; i < real_axis; i++) {
      repeat *= value.Shape()[i];
    }
    std::vector<int64_t> slice;
    TensorShape shape = value.Shape();
    std::vector<TensorShape> shapes;
    for (auto&& item : dims) {
      slice.push_back(item * slice_size);
      shape.Set(real_axis, item);
      shapes.push_back(shape);
    }
    if (repeat == 0 || slice_size == 0) {
      for (int64_t i = 0; i < num_; i++) {
        Tensor output;
        XDL_CHECK_STATUS(ctx->AllocateOutput(i, shapes[i], &output));
      }
    } else if (repeat == 1) {
      Buffer* parent = value.GetBuffer();
      char* ptr = (char*)parent->begin();
      for (int64_t i = 0; i < num_; i++) {
        Buffer* buf = new Buffer(ptr, slice[i], parent);
        XDL_CHECK_STATUS(ctx->SetOutput(i, Tensor(shapes[i], dtype_, buf)));
        buf->UnRef();
        ptr += slice[i];
      }
    } else {
      char* ptr = value.Raw<char>();
      std::vector<char*> ptrs;
      for (int64_t i = 0; i < num_; i++) {
        Tensor output;
        XDL_CHECK_STATUS(ctx->AllocateOutput(i, shapes[i], &output));
        ptrs.push_back(output.Raw<char>());
      }
      for (int64_t k = 0; k < repeat; k++) {
        for (int64_t i = 0; i < num_; i++) {
          memcpy(ptrs[i], ptr, slice[i]);
          ptr += slice[i];
          ptrs[i] += slice[i];
        }
      }
    }
    return Status::Ok();
  }
 private:
  int64_t axis_;
  int64_t num_;
  DataType dtype_;
};

XDL_DEFINE_OP(Split)
  .Input("value", "dtype")
  .Input("num_or_size_splits", DataType::kInt64)
  .Attr("axis", AttrValue::kInt)
  .Attr("num", AttrValue::kInt)
  .Attr("dtype", AttrValue::kDataType)
  .OutputList("result", "dtype", "num");

XDL_REGISTER_KERNEL(Split, SplitOp)
  .Device("CPU");

} // namespace xdl


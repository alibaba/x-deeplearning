/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "xdl/core/ops/add_sparse_gradient.h"

#include "xdl/core/framework/op_define.h"

namespace xdl {


template <typename T, typename I>
class SparseGradAddOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    std::vector<Tensor> in_grads, in_ids;
    XDL_CHECK_STATUS(ctx->GetInputList("in_grads", &in_grads));
    XDL_CHECK_STATUS(ctx->GetInputList("in_ids", &in_ids));
    XDL_CHECK_COND(in_grads.size() == in_ids.size(),
                   Status::ArgumentError("grads and ids size not equal"));
    for (size_t i = 0; i < in_grads.size(); ++i) {
      XDL_CHECK_COND(in_grads[i].Shape().Size() == 2,
                     Status::ArgumentError("grad input dim must be 2"));
      XDL_CHECK_COND(in_ids[i].Shape().Size() == 1 ||
                     in_ids[i].Shape().Size() == 2,
                     Status::ArgumentError("id input dim must be 1 or 2"));
      XDL_CHECK_COND(in_grads[i].Shape()[0] == in_ids[i].Shape()[0],
                     Status::ArgumentError("grad dim 0 not equal to id dim 0"));
    }
    if (in_grads.empty()) return Status::Ok();

    Tensor out_grads, out_ids;
    HostAddSparse<T, I>(in_grads, in_ids, &out_grads, &out_ids);

    ctx->SetOutput(0, out_grads);
    ctx->SetOutput(1, out_ids);

    return Status::Ok();
  }
};

XDL_DEFINE_OP(SparseGradAddOp)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType)
  .Attr("size", AttrValue::kInt)
  .InputList("in_grads", "dtype", "size")
  .InputList("in_ids", "itype", "size")
  .Output("out_grads", "dtype")
  .Output("out_ids", "itype");

#define REGISTER_KERNEL(T, I)                                 \
  XDL_REGISTER_KERNEL(SparseGradAddOp, SparseGradAddOp<T, I>) \
  .Device("CPU")                                              \
  .AttrDataType<T>("dtype")                                   \
  .AttrDataType<I>("itype");

REGISTER_KERNEL(float, int32_t);
REGISTER_KERNEL(float, int64_t);
REGISTER_KERNEL(double, int32_t);
REGISTER_KERNEL(double, int64_t);

#undef REGISTER_KERNEL

}  // namespace xdl

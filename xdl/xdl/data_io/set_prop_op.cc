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


#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/data_io/data_io.h"
#include "xdl/data_io/pool.h"

namespace xdl {

template <typename T>
class SetPropOp: public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    std::string ds;
    XDL_CHECK_STATUS(ctx->GetAttr("ds", &ds));

    data_io_ = io::DataIOMap::Instance()->Get(ds);
    XDL_CHECK(data_io_ != nullptr);

    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor prop;
    XDL_CHECK_STATUS(ctx->GetInput(0, &prop));

    auto batch = data_io_->CurrBatch();
    XDL_CHECK(batch != nullptr);

    auto props = prop.Raw<float>();
    auto dims = prop.Shape().Dims();

    auto sgroups = batch->sgroups(); 
    XDL_CHECK(sgroups.size() > 0);

    int i = 0;
    for (auto &sgroup: sgroups) {
      auto sg = sgroup->Get();
      XDL_CHECK(sg != nullptr);
      for (int j = sgroup->begin_; j < sgroup->end_; ++i, ++j) {
        XDL_CHECK(i < dims[0] && j < sg->labels_size());
        if (sg->props_size() == j) {
          auto p = sg->add_props();
          for (int k = 0; k < dims[1]; ++k) {
            p->add_values(props[i*dims[1] + k]);
          }
        } else {
          XDL_CHECK(sg->props_size() == sg->labels_size());
          auto p = sg->mutable_props(j);
          XDL_CHECK(p->values_size() == dims[1]);
          for (int k = 0; k < dims[1]; ++k) {
            p->set_values(k, props[i*dims[1] + k]);
          }
        }
      }
    }
    return Status::Ok();
  }

 private:
  io::DataIO *data_io_;
};

XDL_DEFINE_OP(SetProp)
  .Attr("ds", AttrValue::kString)
  .Attr("dtype", AttrValue::kDataType)
  .Input("prop", "dtype");

#define REGISTER_KERNEL(T)                   \
  XDL_REGISTER_KERNEL(SetProp, SetPropOp<T>) \
  .Device("CPU")                             \
  .AttrDataType<T>("dtype")

REGISTER_KERNEL(int32_t);
REGISTER_KERNEL(int64_t);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

}  // namespace xdl

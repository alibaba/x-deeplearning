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

#include "xdl/core/utils/logging.h"

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"

namespace xdl {

namespace {

template <typename T>
T GetNonClick(T* plabels, size_t k, int dim) {
  if (dim == 1) return 1.0 - plabels[k];
  return plabels[2 * k];
}

template <typename T>
T GetClick(T* plabels, size_t k, int dim) {
  if (dim == 1) return plabels[k];
  return plabels[2 * k + 1];
}

template <typename T>
bool ComputeGauc(T* plabels, T* ppreds, T* pfilter, size_t* pidx,
                 size_t l, size_t r, int dim, double* ret) {
  std::sort(pidx + l, pidx + r, [ppreds, dim](size_t a, size_t b) {
    return GetClick<T>(ppreds, a, dim) < GetClick<T>(ppreds, b, dim);
  });
  double fp1, tp1, fp2, tp2, auc;
  fp1 = tp1 = fp2 = tp2 = auc = 0;
  size_t i;
  for (size_t k = l; k < r; ++k) {
    i = pidx[k];
    if (pfilter != nullptr && pfilter[i] == 0) continue;
    fp2 += GetNonClick<T>(plabels, i, dim);
    tp2 += GetClick<T>(plabels, i, dim);
    auc += (fp2 - fp1) * (tp2 + tp1);
    fp1 = fp2;
    tp1 = tp2;
  }
  double threshold = static_cast<double>(r - l) - 1e-3;
  if (tp2 > threshold or fp2 > threshold) {
    *ret = -0.5;
    return true;
  }
  if (tp2 * fp2 > 0) {
    *ret = (1.0 - auc / (2.0 * tp2 * fp2));
    return true;
  }
  return false;
}

}  // namespace

template <typename T, typename I>
class GaucCalcOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor labels, predicts, indicator, filter, gauc, pv_num;
    XDL_CHECK_STATUS(ctx->GetInput(0, &labels));
    XDL_CHECK_STATUS(ctx->GetInput(1, &predicts));
    XDL_CHECK_STATUS(ctx->GetInput(2, &indicator));
    XDL_CHECK_STATUS(ctx->GetInput(3, &filter));
    size_t ldim = labels.Shape().Size();
    size_t pdim = predicts.Shape().Size();
    size_t n = labels.Shape()[0];
    XDL_CHECK(ldim == pdim && n == predicts.Shape()[0])
        << "labels and indicator must have same dim";
    XDL_CHECK(pdim == 1 || pdim == 2) << "predictions must be 1 or 2 dim";
    XDL_CHECK(n == indicator.Shape().NumElements())
        << "labels and indicator must have same number of elements";
    if (ldim == 2) {
      XDL_CHECK(labels.Shape()[1] == 2 && predicts.Shape()[1] == 2)
          << "second dim must be 2";
    }
    T* pfilter = nullptr;
    if (filter.Shape().NumElements() != 0) {
      XDL_CHECK(filter.Shape().NumElements() == n) << "invalid filter shape";
      pfilter = filter.Raw<T>();
    }
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape({}), &gauc));
    XDL_CHECK_STATUS(ctx->AllocateOutput(1, TensorShape({}), &pv_num));
    *(gauc.Raw<double>()) = 0;
    *(pv_num.Raw<int64_t>()) = 0;

    T* plabels = labels.Raw<T>(), *ppreds = predicts.Raw<T>();
    I* pind = indicator.Raw<I>();
    std::vector<size_t> index(n);
    for (size_t i = 0; i < n; ++i) {
      index[i] = i;
    }
    size_t* pidx = index.data();
    size_t begin = 0;
    bool first = true;
    for (size_t end = 0; end < n; ++end) {
      if (pind[end] != pind[begin]) {
        if (first) {
          first = false;
        } else {
          double auc = 0;
          if (ComputeGauc<T>(plabels, ppreds, pfilter, pidx, begin, end, ldim, &auc)) {
              if (auc >= 0) {
                *(gauc.Raw<double>()) += auc * (end - begin);
                *(pv_num.Raw<int64_t>()) += (end - begin);
              }
          }
        }
        begin = end;
      }
    }
    return Status::Ok();
  }
};

XDL_DEFINE_OP(GaucCalcOp)
  .Input("labels", "dtype")
  .Input("predicts", "dtype")
  .Input("indicator", "itype")
  .Input("filter", "dtype")
  .Output("gauc", DataType::kDouble)
  .Output("pv_num", DataType::kInt64)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType);

#define REGISTER_KERNEL(T, I)                       \
  XDL_REGISTER_KERNEL(GaucCalcOp, GaucCalcOp<T, I>) \
  .Device("CPU")                                    \
  .AttrDataType<T>("dtype")                         \
  .AttrDataType<I>("itype")

REGISTER_KERNEL(float, int32_t);
REGISTER_KERNEL(float, int64_t);
REGISTER_KERNEL(double, int32_t);
REGISTER_KERNEL(double, int64_t);

#undef REGISTER_KERNEL

}  // namespace xdl

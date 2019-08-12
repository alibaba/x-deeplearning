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

#include "xdl/core/ops/merge_sparse_op.h"

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"

namespace xdl {

template <typename T, typename V, typename I>
Status MergeSparseOp<T, V, I>::Init(OpKernelConstruction* ctx) {
  return Status::Ok();
}

template <typename T, typename V, typename I>
Status MergeSparseOp<T, V, I>::Compute(OpKernelContext* ctx) {
  std::vector<Tensor> id_list, value_list, segment_list;
  Tensor ids, values, segments, groups;
  XDL_CHECK_STATUS(ctx->GetInputList("id_list", &id_list));
  XDL_CHECK_STATUS(ctx->GetInputList("value_list", &value_list));
  XDL_CHECK_STATUS(ctx->GetInputList("segment_list", &segment_list));
  XDL_CHECK_COND(id_list.size() == value_list.size() &&
                 id_list.size() == segment_list.size(),
                 Status::ArgumentError("input list size not equal"));
  for (size_t i = 1; i < segment_list.size(); ++i) {
    XDL_CHECK_COND(segment_list[i].Shape().NumElements() ==
                   segment_list[i-1].Shape().NumElements(),
                   Status::ArgumentError("merged sample size must be equal"));
  }
  
  size_t id_size = 0, value_size = 0, segment_size = segment_list[0].Shape()[0];
  for (size_t i = 0; i < id_list.size(); ++i) {
    XDL_CHECK_COND(id_list[i].Shape()[0] == value_list[i].Shape()[0] ||
                   value_list[i].Shape().NumElements() == 0,
                   Status::ArgumentError("id and value size must be equal"));
    id_size += id_list[i].Shape()[0];
    value_size += value_list[i].Shape()[0];
  }
  std::vector<V*> value_ptr(value_list.size());
  for (size_t i = 0; i < value_list.size(); ++i) {
    if (value_list[i].Shape().NumElements() == 0) {
      XDL_CHECK(value_size == 0) << "must be all empty values";
      value_ptr[i] = nullptr;
    } else {
      value_ptr[i] = value_list[i].Raw<V>();
    }
  }
  TensorShape id_shape(id_list[0].Shape().Dims());
  id_shape.Set(0, id_size);
  TensorShape value_shape({value_size});
  TensorShape segment_shape({segment_size});
  size_t group_size = id_list.size();
  TensorShape group_shape({group_size * segment_size});

  size_t id_num = id_list[0].Shape().NumElements() / id_list[0].Shape()[0];

  XDL_CHECK_STATUS(ctx->AllocateOutput(0, id_shape, &ids));
  XDL_CHECK_STATUS(ctx->AllocateOutput(1, value_shape, &values));
  XDL_CHECK_STATUS(ctx->AllocateOutput(2, segment_shape, &segments));
  XDL_CHECK_STATUS(ctx->AllocateOutput(3, group_shape, &groups));

  T* pid = ids.Raw<T>();
  V* pvalue = values.Raw<V>();
  I* pseg = segments.Raw<I>();
  I* pgrp = groups.Raw<I>();

  I begin = 0, end = 0;
  I grp_cnt = 0;
  for (size_t i = 0; i < segment_size; ++i) {
    for (size_t j = 0; j < group_size; ++j) {
      I* seg = segment_list[j].Raw<I>();
      begin = i == 0 ? 0 : seg[i-1];
      end = seg[i];
      T* id = id_list[j].Raw<T>();
      V* value = value_ptr[j];
      for (I k = begin; k < end; ++k) {
        for (size_t l = 0; l < id_num; ++l) {
          *pid++ = id[k * id_num + l];
        }
        if (value) *pvalue++ = value[k];
      }
      grp_cnt += (end - begin);
      *pgrp++ = grp_cnt;
    }
    *pseg++ = grp_cnt;
  }
  return Status::Ok();
}

XDL_DEFINE_OP(MergeSparseOp)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("vtype", AttrValue::kDataType)
  .Attr("itype", AttrValue::kDataType)
  .Attr("size", AttrValue::kInt)
  .InputList("id_list", "dtype", "size")
  .InputList("value_list", "vtype", "size")
  .InputList("segment_list", "itype", "size")
  .Output("ids", "dtype")
  .Output("values", "vtype")
  .Output("segments", "itype")
  .Output("groups", "itype");

#define REGISTER_KERNEL(T, V, I)                             \
  XDL_REGISTER_KERNEL(MergeSparseOp, MergeSparseOp<T, V, I>) \
  .Device("CPU")                                             \
  .AttrDataType<T>("dtype")                                  \
  .AttrDataType<V>("vtype")                                  \
  .AttrDataType<I>("itype");

REGISTER_KERNEL(int32_t, float, int32_t);
REGISTER_KERNEL(int64_t, float, int32_t);

#undef REGISTER_KERNEL

}  // namespace xdl

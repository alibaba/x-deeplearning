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
#ifndef XDL_CORE_OPS_TILE_OP_H_
#define XDL_CORE_OPS_TILE_OP_H_

#include "xdl/core/framework/op_kernel.h"

namespace xdl {

template <typename T, typename I>
class TileOp : public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override;
  Status Compute(OpKernelContext* ctx) override;
 private:
  bool reverse_;
  int64_t length_;
};

}  // namespace xdl

#endif  // XDL_CORE_OPS_TILE_OP_H_
